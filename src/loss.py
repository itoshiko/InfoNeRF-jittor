import numpy as np
import jittor as jt
import jittor.nn as nn

from jt_ssim import ssim, ms_ssim
from jt_lpips import LPIPS

# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.Var([10.]))
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def img2psnr_redefine(x, y):
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)
    
    [new]
    average PSNR(each image pair)
    '''
    image_num = x.size(0)
    mses = ((x - y) ** 2).reshape(image_num, -1).mean(-1)
    
    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.stack(psnrs).mean()
    return psnr

def img2psnr_mask(x, y, mask):
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)
    
    [new]
    average PSNR(each image pair)
    '''
    image_num = x.size(0)
    mses = ((x-y)**2).mean(-1)
    mses_sum = (mses*mask).reshape(image_num, -1).sum(-1)
    mses = mses_sum /mask.reshape(image_num, -1).sum(-1)
    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.mean(jt.stack(psnrs))
    return psnr

def img2ssim(x, y, mask=None):
    if mask is not None:
        x = mask.unsqueeze(-1)*x
        y = mask.unsqueeze(-1)*y
     
    ssim_ = ssim(x, y, data_range=1)
    ms_ssim_ = ms_ssim(x, y, data_range=1)
    return ssim_, ms_ssim_

def img2lpips(x, y, net='vgg'):
    loss_fn = LPIPS(net=net, spatial=False)
    return loss_fn.execute(x, y)

######################################################
#           Ray Entropy Minimization Loss            #
######################################################

class EntropyLoss:
    def __init__(self, args):
        super(EntropyLoss, self).__init__()
        self.type_ = args['entropy_type'] 
        self.threshold = args['entropy_acc_threshold']
        self.computing_entropy_all = args['computing_entropy_all']
        self.computing_ignore_smoothing = args['computing_entropy_all']
        self.entropy_log_scaling = args['computing_entropy_all']
        self.N_entropy = args['N_entropy'] 
        
        if self.N_entropy == 0:
            self.computing_entropy_all = True
    
    def ray(self, density, acc):
        density = nn.relu(density[..., -1])
        sigma = 1 - jt.exp(-density)
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1) + 1e-10)
        entropy_ray = jt.sum(self.entropy(ray_prob), -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray*= mask
        entropy_ray_loss = jt.mean(entropy_ray, -1)
        if self.entropy_log_scaling:
            return jt.log(entropy_ray_loss + 1e-10)
        return entropy_ray_loss

    def ray_zvals(self, sigma, acc):
        ray_prob = sigma / (jt.sum(sigma,-1).unsqueeze(-1) + 1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc > self.threshold).stop_grad()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)
    
    def ray_zvals_ver1_sigma(self, sigma, dists, acc):
        ray_prob = sigma / (jt.sum(sigma* dists,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        
        #intergral
        entropy_ray = entropy_ray * dists
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def ray_zvals_ver2_alpha(self, alpha, dists, acc):
        ray_prob = alpha / (jt.sum(alpha,-1).unsqueeze(-1)+1e-10)
        
        entropy_ray = -1 * ray_prob * jt.log2(ray_prob/(dists+1e-10)+1e-10)
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)
    
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1 * prob * jt.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob * jt.log2(1-prob)


######################################################
#          Infomation Gain Reduction Loss            #
######################################################

class SmoothingLoss:
    def __init__(self, args):
        super(SmoothingLoss, self).__init__()
        self.smoothing_activation = args['activation']
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def __call__(self, sigma_1, sigma_2):
        if self.smoothing_activation == 'softmax':
            p = nn.softmax(sigma_1, -1)
            q = nn.softmax(sigma_2, -1)
        elif self.smoothing_activation == 'norm':
            p = sigma_1 / (jt.sum(sigma_1, -1, keepdims=True) + 1e-10) + 1e-10
            q = sigma_2 / (jt.sum(sigma_2, -1, keepdims=True) + 1e-10) + 1e-10
        loss = self.criterion(p.log(), q)
        # pointwise =  q * (q.log() - p.log())
        # return jt.sum(pointwise) / pointwise.shape[0]
        return loss
