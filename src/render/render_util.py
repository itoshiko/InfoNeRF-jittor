import jittor as jt
import jittor.nn as nn
import numpy as np


def normalize_pts(pts, H, W):
    pts[..., 0] = pts[..., 0] / W
    pts[..., 1] = pts[..., 1] / H
    pts = pts * 2.0 - 1.0
    return pts


def get_rays(H, W, focal, c2w, padding=None):
    """get rays in world coordinate of full image

    Args:
        H (int): height of target image
        W (int): width of target image
        focal (flaot): focal length of camera
        c2w (jittor.Var): transform matrix from camera coordinate to world coordinate
        padding (int, optional): padding border around the image. Defaults to None.

    Returns:
        tuple(jittor.Var[H, W, 3], jittor.Var[H, W, 3]): 
        origin of rays, direction of rays, both transformed to world coordinate
    """
    # create pts in pixel coordinate
    if padding is not None:
        i, j = jt.meshgrid(jt.linspace(-padding, W-1+padding, W+2*padding), jt.linspace(-padding, H-1+padding, H+2*padding)) 
    else:
        i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = jt.transpose(i, (1, 0))
    j = jt.transpose(j, (1, 0))
    # transform to camera coordinate
    dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1)  # [H, W, 3]
    # Rotate ray directions from camera frame to the world frame
    rays_d =  dirs @ jt.transpose(c2w[:3, :3], (1, 0))
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].repeat((H, W, 1))  # [H, W, 3]
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Get rays in Normalized Device Coordinates (NDC), a [-1, 1]^3 cube.
    Our goal is to take a ray o+td and calculate a ray origin o' and 
    direction d' in NDC space such that for every t, there exists a t' 
    such that π(o+td) = o'+t'd' (π is projection using the above matrix). 
    In other words, the projected original ray and the NDC space ray 
    trace out the same points (but not necessarily at the same rate).
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def random_sample_ray(H, W, focal, c2w, cnt, pix_coord=None, center_crop=1.0):
    """Sample rays in target view

    Args:
        H (int): image height
        W (int): image width
        focal (float): focal length
        c2w (jt.Var): transform matrix from camera to world
        cnt (int): number of sample points
        pix_coord (jt.Var, optional): pixel coords of sample points. If None, random 
        points will be sampled. Defaults to None.
        center_crop (float, optional): sample in center area. Defaults to 1.0.

    Returns:
        tuple: rays_o ([cnt, 3]), rays_d ([cnt, 3]), 
        pix_coord ([cnt, 2], in pixel coords, unnormalized)
    """
    # generate pixel coord
    if pix_coord is None:
        sample_x = jt.rand(cnt, dtype=jt.float32) * W * center_crop
        sample_y = jt.rand(cnt, dtype=jt.float32) * H * center_crop
        sample_x += ((1 - center_crop) / 2.) * W
        sample_y += ((1 - center_crop) / 2.) * H  # [N, ]
        pix_coord = jt.stack([sample_x, sample_y], dim=-1)
    else:
        sample_x = pix_coord[..., 0]
        sample_y = pix_coord[..., 1]

    dirs = jt.stack([(sample_x-W*.5)/focal, -(sample_y-H*.5)/focal, -jt.ones_like(sample_x)], -1)  # [N, 3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.matmul(dirs, jt.transpose(c2w[:3, :3], (1, 0)))
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].unsqueeze(0).repeat((dirs.shape[0], 1))  # [N, 3]
    return rays_o, rays_d, pix_coord
                

def sample_nearby_ray(H, W, focal, c2w, pix_coord, distance):
    new_x, new_y = pix_coord[..., 0], pix_coord[..., 1]
    pts_shape = pix_coord.shape[:-1]
    offset = np.random.randint(1, distance * 2 + 1)
    offset_x = jt.randint(0, offset + 1, pts_shape)
    offset_y = offset - offset_x
    
    new_x += offset_x
    new_y += offset_y
    new_pix_coord = jt.stack([new_x, new_y], dim=-1)

    dirs = jt.stack([(new_x-W*.5) / focal, -(new_y-H*.5) / focal, -jt.ones_like(new_x)], -1)  # [..., 3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.matmul(dirs, c2w[:3, :3])
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].repeat(list(dirs.shape[:-1]) + [1, ])  # [..., 3]
    return rays_o, rays_d, new_pix_coord


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    # u = u.contiguous()
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = jt.where(denom<1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def raw2alpha(raw, dists, act_fn=nn.relu):
    return 1. - jt.exp(-act_fn(raw) * dists)


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False,
               out_alpha=False, out_sigma=False, out_dist=False, debug_save=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """    
    # distance between sample points
    dists = z_vals[..., 1:] - z_vals[..., :-1] 
    dists = jt.concat([dists, jt.expand(jt.array([1e10]), dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * jt.norm(rays_d, dim=-1, keepdims=True)  # [N_rays, N_samples]

    rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.randn(raw[..., 3].shape) * raw_noise_std

    # alpha: 1. - exp(-delta * sigma)
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    sigma = nn.relu(raw[..., 3] + noise)  # for sigma output
    # transmission, integral(sigma * delta) * alpha
    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # do the integral to obtain rgb output
    rgb_map = jt.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1. / jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[...,None])
    
    output = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map':acc_map, 'weights': weights, 'depth_map': depth_map}
    if out_alpha:
        output.update({'alpha': alpha})
    if out_sigma:
        output.update({'sigma': sigma})
    if out_dist:
        output.update({'dists': dists})
    return output


def sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query):
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query(pts, viewdirs, network)

    rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    sigma = nn.relu(raw[..., 3])

    output = raw2outputs(raw, z_vals, rays_d)
    return rgb, sigma, output['depth_map']


def generate_pts(rays_o, rays_d, near, far, N_samples, lindisp=False, perturb=False):
    near, far = near * jt.ones_like(rays_d[..., :1]), far * jt.ones_like(rays_d[..., :1])

    t_vals = jt.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # not sample at fixed z, but random position betwween [z, z+1]
    if perturb:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.concat([mids, z_vals[..., -1:]], -1)
        lower = jt.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]
    return pts, z_vals


