import os
import numpy as np
import time
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm, trange

from config import load_config
from dataset.load_blender import load_blender_data_ex
import train_util as tu
import render.render_util as ru
import render.nerf as nf
import loss as lfn


class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        data_cfg = self.cfg['dataset']
        train_cfg = self.cfg['training']
        data_cfg.update(train_cfg)
        self.loaded_data = load_blender_data_ex(data_cfg)
        self.img_h, self.img_w, self.focal = self.loaded_data['calib']
        print(f"Loaded blender: {data_cfg['datadir']}")
        self.exp_path = tu.prepare_dir(cfg)
        print('Exp path: ', self.exp_path)
        self.model, self.model_fine, self.embed_fn, self.embeddirs_fn = tu.create_nerf(cfg)
        self.get_near_c2w = tu.GetNearC2W(cfg['info_loss'])
        self.loss_fn = {}
        self.init_loss()
        op_param = list(self.model.parameters()) + (list(self.model_fine.parameters()) if self.model_fine is not None else [])
        self.optimizer = jt.optim.Adam(params=op_param, lr=self.cfg['training']['lr'], betas=(0.9, 0.999))
        self.it_time = 0

    def init_loss(self):
        if self.cfg['entropy_loss']['use']:
            self.loss_fn['ent'] = lfn.EntropyLoss(self.cfg['entropy_loss'])
        if self.cfg['info_loss']['use']:
            self.loss_fn['kl_smooth'] = lfn.SmoothingLoss(self.cfg['info_loss'])
        self.loss_fn['img_loss'] = lfn.img2mse

    def gen_train_data(self):
        sample_info_gain = self.cfg['info_loss']['use']
        sample_entropy = self.cfg['entropy_loss']['use']
        loaded_ray = {}

        # Random from one image
        i_train = self.loaded_data['i_split'][0]
        img_i = np.random.choice(i_train)
        target = self.loaded_data['imgs'][img_i:img_i+1]  # [1, 3, H, W]
        rgb_pose = self.loaded_data['poses'][img_i]  # [4, 4]

        cfg_train = self.cfg['training']
        _crop = cfg_train['precrop_frac'] if self.it_time < cfg_train['precrop_iters'] else 1.0
        # sample rays in target view, ([N_rand, 3], [N_rand, 3], [N_rand, 2])
        rays_o, rays_d, coord = ru.random_sample_ray(
            self.img_h, self.img_w, self.focal, rgb_pose, 
            self.cfg['training']['N_rand'], center_crop=_crop)
        # sample ground truth RGB
        coord = ru.normalize_pts(coord, self.img_h, self.img_w)  # normalize to [-1, 1] for sampling
        coord = coord.unsqueeze(0).unsqueeze(0)  # [1, 1, N_rand, 2]
        target_rgb = nn.grid_sampler_2d(target, coord, 'bilinear', 'zeros', False)  # [1, 3, 1, N_rand]
        target_rgb = target_rgb.squeeze(0).squeeze(-2)
        target_rgb = target_rgb.permute(1, 0)  # [N_rand, 3]
        coord = coord.squeeze(0).squeeze(0)
        
        loaded_ray.update({'coord': coord, 'rays_o': rays_o, 'rays_d': rays_d, 'target_rgb': target_rgb})

        # sample rays for information gain reduction loss
        if sample_info_gain:
            if self.cfg['info_loss']['sampling_method'] == 'near_pose':
                rgb_near_pose = self.get_near_c2w(rgb_pose)
                # sample rays in a nearby view for info gain loss, 
                # ([N_rand, 3], [N_rand, 3], [N_rand, 2])
                rays_o_near, rays_d_near, _ = ru.random_sample_ray(
                    self.img_h, self.img_w, self.focal, rgb_near_pose, 
                    self.cfg['training']['N_rand'], pix_coord=coord)
            else:
                rays_o_near, rays_d_near, _ = ru.sample_nearby_ray(
                    self.img_h, self.img_w, self.focal, rgb_pose, 
                    coord, self.cfg['info_loss']['pixel_range'])
            loaded_ray.update({'rays_o_near': rays_o_near, 'rays_d_near': rays_d_near})

        # Sampling for unseen rays
        if sample_entropy:
            n_entropy = self.cfg['entropy_loss']['N_entropy']
            # randomly choose from all images
            img_e = np.random.choice(self.loaded_data['imgs'].shape[0])
            rgb_pose_e = self.loaded_data['poses'][img_e]  # [4, 4]
            rays_o_ent, rays_d_ent, coord_ent = ru.random_sample_ray(
                self.img_h, self.img_w, self.focal, rgb_pose_e, 
                n_entropy, center_crop=_crop)
            loaded_ray.update({'rays_o_ent': rays_o_ent, 'rays_d_ent': rays_d_ent})

            # sample rays for information gain reduction loss (for unseen rays)
            if sample_info_gain:
                if self.cfg['info_loss']['sampling_method'] == 'near_pose':
                    ent_near_pose = self.get_near_c2w(rgb_pose_e)
                    # sample rays in a nearby view for info gain loss, 
                    # ([N_rand, 3], [N_rand, 3], [N_rand, 2])
                    rays_o_ent_near, rays_d_ent_near, _ = ru.random_sample_ray(
                        self.img_h, self.img_w, self.focal, ent_near_pose, 
                        n_entropy, pix_coord=coord_ent)
                else:
                    rays_o_ent_near, rays_d_ent_near, _ = ru.sample_nearby_ray(
                        self.img_h, self.img_w, self.focal, rgb_pose_e, 
                        coord_ent, self.cfg['info_loss']['pixel_range'])
                loaded_ray.update({'rays_o_ent_near': rays_o_ent_near, 'rays_d_ent_near': rays_d_ent_near})
        
        return loaded_ray

    def render_rays(self, rays_o, rays_d, N_samples, N_importance=0, eval=False):
        """Volumetric rendering for rays
        """
        result = {}
        # calculate samplr points along rays
        near, far = self.cfg['rendering']['near'], self.cfg['rendering']['far']
        pts, z_vals = ru.generate_pts(
            rays_o, rays_d, near, far, N_samples, 
            perturb=((not eval) and (self.cfg['rendering']['perturb'])))
        if self.cfg['rendering']['use_viewdirs']:
            view_dirs = rays_d / jt.norm(rays_d, dim=-1, keepdims=True)
            view_dirs = view_dirs.unsqueeze(1)
            view_dirs = jt.repeat(view_dirs, (1, N_samples, 1))  # [N_rays, N_samples, 3]
        else:
            view_dirs = None
        raw_out = nf.run_network(
            pts, view_dirs, self.model, self.embed_fn, self.embeddirs_fn, 
            self.cfg['training']['netchunk'])
        decoded_out = ru.raw2outputs(
            raw_out, z_vals, rays_d, 
            self.cfg['rendering']['raw_noise_std'], self.cfg['dataset']['white_bkgd'],
            out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))
        result.update({'coarse': decoded_out})

        if N_importance > 0:
            with jt.no_grad():
                weights = decoded_out['weights']
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = ru.sample_pdf(
                    z_vals_mid, weights[...,1:-1], N_importance, 
                    det=((not eval) and (self.cfg['rendering']['perturb'])))
                z_samples = z_samples.detach()
                _, z_vals_re = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)
            
            # [N_rays, N_samples + N_importance, 3]
            pts_re = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_re[..., :, None]
            if view_dirs is not None:
                # [N_rays, N_samples + N_importance, 3]
                view_dirs_re = jt.repeat(view_dirs[:, :1, :], (1, N_samples + N_importance, 1))
            raw_re = nf.run_network(
                pts_re, view_dirs_re, 
                self.model_fine if self.model_fine is not None else self.model, 
                self.embed_fn, self.embeddirs_fn, 
                self.cfg['training']['netchunk'])
            decoded_out_re = ru.raw2outputs(
                raw_re, z_vals_re, rays_d, 
                self.cfg['rendering']['raw_noise_std'], self.cfg['dataset']['white_bkgd'],
                out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))
            result.update({'fine': decoded_out_re})

        return result

    def train(self):
        N_sample = self.cfg['rendering']['N_samples']
        N_refine = self.cfg['rendering']['N_importance']
        N_rays = self.cfg['training']['N_rand']
        N_entropy = self.cfg['entropy_loss']['N_entropy']
        for it in tqdm(range(self.cfg['training']['N_iters'])):
            self.it_time = it
            self.optimizer.zero_grad()
            train_data = self.gen_train_data()
            all_rays_o = []
            all_rays_d = []
            for _entry in ['rays_o', 'rays_o_near', 'rays_o_ent', 'rays_o_ent_near']:
                if _entry in train_data:
                    all_rays_o.append(train_data[_entry])
            for _entry in ['rays_d', 'rays_d_near', 'rays_d_ent', 'rays_d_ent_near']:
                if _entry in train_data:
                    all_rays_d.append(train_data[_entry])
            all_rays_o = jt.concat(all_rays_o, dim=0)  # [N, 3]
            all_rays_d = jt.concat(all_rays_d, dim=0)  # [N, 3]

            render_out = self.render_rays(all_rays_o, all_rays_d, N_sample, N_refine)

            total_loss = 0.
            loss_dict = {}
            # image loss
            gt_rgb = train_data['target_rgb']
            predict_rgb = render_out['coarse']['rgb_map'][:N_rays]
            img_loss = self.loss_fn['img_loss'](gt_rgb, predict_rgb)
            loss_dict['img_loss'] = img_loss.item()
            total_loss += img_loss
            if 'fine' in render_out:
                predict_rgb_fine = render_out['fine']['rgb_map'][:N_rays]
                img_loss_fine = self.loss_fn['img_loss'](gt_rgb, predict_rgb_fine)
                loss_dict['img_loss_fine'] = img_loss_fine.item()
                total_loss += img_loss_fine
            
            # Ray Entropy Minimiation Loss
            ent_iter = (it < self.cfg['entropy_loss']['entropy_end_iter']) if self.cfg['entropy_loss']['entropy_end_iter'] > 0 else True
            if self.cfg['entropy_loss']['use'] and ent_iter:
                alpha_raw = render_out['fine']['alpha'] \
                    if 'fine' in render_out else render_out['coarse']['alpha']
                acc_raw = render_out['fine']['acc_map'] \
                    if 'fine' in render_out else render_out['coarse']['acc_map']

                need_remove_nearby = self.cfg['info_loss']['use'] and self.cfg['entropy_loss']['entropy_ignore_smoothing']
                near_remove_normal = (not self.cfg['entropy_loss']['computing_entropy_all']) or self.cfg['entropy_loss']['N_entropy'] <= 0
                # only compute loss for rays + unseen rays, no nearby rays
                if need_remove_nearby:
                    alpha_raw = jt.concat([alpha_raw[:N_rays], alpha_raw[2*N_rays:2*N_rays+N_entropy]], dim=0)
                    acc_raw = jt.concat([acc_raw[:N_rays], acc_raw[2*N_rays:2*N_rays+N_entropy]], dim=0)
                    if near_remove_normal:
                        alpha_raw = alpha_raw[self.N_rays:]
                        sigma = sigma[self.N_rays:]
                elif self.cfg['info_loss']['use']:
                    if near_remove_normal:
                        alpha_raw = alpha_raw[2*N_rays:]
                        acc_raw = acc_raw[2*N_rays:]
                elif near_remove_normal:
                    alpha_raw = alpha_raw[N_rays:]
                    acc_raw = acc_raw[N_rays:]

                entropy_ray_zvals_loss = self.loss_fn['ent'].ray_zvals(alpha_raw, acc_raw)
                loss_dict['entropy_ray_zvals'] = entropy_ray_zvals_loss.item()
                total_loss += self.cfg['entropy_loss']['entropy_ray_zvals_lambda'] * entropy_ray_zvals_loss

            # Infomation Gain Reduction Loss
            info_iter = (it < self.cfg['info_loss']['info_end_iter']) if self.cfg['info_loss']['info_end_iter'] > 0 else True
            if self.cfg['info_loss']['use'] and info_iter:
                alpha_raw = render_out['fine']['alpha'] \
                    if 'fine' in render_out else render_out['coarse']['alpha']
                info_loss = self.loss_fn['kl_smooth'](alpha_raw[:N_rays], alpha_raw[N_rays:2*N_rays])
                if self.cfg['entropy_loss']['use']:
                    info_loss += self.loss_fn['kl_smooth'](alpha_raw[2*N_rays:2*N_rays+N_entropy], alpha_raw[2*N_rays+N_entropy:])
                loss_dict['KL_loss'] = info_loss.item()
                total_loss += self.cfg['info_loss']['info_lambda'] * info_loss
                
            self.optimizer.step(total_loss)
            if it % 100 == 99:
                print(loss_dict)
        
        jt.save(self.model.state_dict(), os.path.join(self.exp_path, 'ckpt', 'model.pth'))
        if self.model_fine is not None:
            jt.save(self.model.state_dict(), os.path.join(self.exp_path, 'ckpt', 'model_fine.pth'))


if __name__=='__main__':
    jt.flags.use_cuda = 1
    jt.set_global_seed(0)
    train_cfg = load_config('configs/lego.toml', 'configs/base.toml')
    trainer = Trainer(train_cfg)
    trainer.train()
