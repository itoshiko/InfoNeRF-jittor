import os
import toml
import jittor as jt
import numpy as np
import render.nerf as n


class GetNearC2W:
    def __init__(self, args):
        super(GetNearC2W, self).__init__()
        self.near_c2w_type = args['near_c2w_type']
        self.near_c2w_rot = args['near_c2w_rot']
        self.near_c2w_trans = args['near_c2w_trans']
    
    def __call__(self, c2w, all_poses=None, j=None):
        if self.near_c2w_type == 'rot_from_origin':
            return self.rot_from_origin(c2w)
        elif self.near_c2w_type == 'near':
            return self.near(c2w, all_poses)
        elif self.near_c2w_type == 'random_pos':
            return self.random_pos(c2w)
        elif self.near_c2w_type == 'random_dir':
            return self.random_dir(c2w, j)
   
    def random_pos(self, c2w):
        c2w[:3, -1] += self.near_c2w_trans * jt.randn(3)
        return c2w 
    
    def random_dir(self, c2w, j):
        rot_mat = self.get_rotation_matrix(j)
        rot = rot_mat @ c2w[:3,:3]  # [3, 3]
        c2w[:3, :3] = rot
        return c2w
    
    def rot_from_origin(self, c2w):
        rot = c2w[:3, :3]  # [3, 3]
        pos = c2w[:3, -1:]  # [3, 1]
        rot_mat = self.get_rotation_matrix()
        pos = rot_mat @ pos
        rot = rot_mat @ rot
        new_c2w = jt.zeros((4, 4), dtype=jt.float32)
        new_c2w[:3, :3] = rot
        new_c2w[:3, -1:] = pos
        new_c2w[3, 3] = 1
        return new_c2w

    def get_rotation_matrix(self):
        rotation = self.near_c2w_rot

        phi = (rotation*(np.pi / 180.))
        x = np.random.uniform(-phi, phi)
        y = np.random.uniform(-phi, phi)
        z = np.random.uniform(-phi, phi)
        
        rot_x = np.array([
            [1,0,0],
            [0,np.cos(x),-np.sin(x)],
            [0,np.sin(x), np.cos(x)]])
        rot_y = np.array([
            [np.cos(y),0,-np.sin(y)],
            [0,1,0],
            [np.sin(y),0, np.cos(y)]])
        rot_z = np.array([
            [np.cos(z),-np.sin(z),0],
            [np.sin(z),np.cos(z),0],
            [0,0,1]])
        _rot = rot_x @ (rot_y @ rot_z)
        return jt.array(_rot)


def prepare_dir(cfg):
    # Create log dir and copy the config file
    basedir = cfg['basedir']
    expname = cfg['expname']
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'result'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.toml')
    with open(f, 'w') as _f:
        toml.dump(cfg, _f)
    return os.path.join(basedir, expname)


def create_nerf(cfg):
    """Instantiate NeRF's MLP model.
    """
    cfg_ren = cfg['rendering']
    cfg_train = cfg['training']
    embed_fn, input_ch = n.get_embedder(cfg_ren['multires'], cfg_ren['i_embed'])

    input_ch_views = 0
    embeddirs_fn = None
    if cfg['rendering']['use_viewdirs']:
        embeddirs_fn, input_ch_views = n.get_embedder(cfg_ren['multires_views'], cfg_ren['i_embed'])
    output_ch = 5 if cfg_ren['N_importance'] > 0 else 4
    
    model = n.NeRF(
        D=cfg_train['netdepth'], W=cfg_train['netwidth'],
        input_ch=input_ch, output_ch=output_ch, skips=[4, ],
        input_ch_views=input_ch_views, use_viewdirs=cfg['rendering']['use_viewdirs'])

    model_fine = None
    if cfg_ren['N_importance'] > 0:
        model_fine = n.NeRF(
            D=cfg_train['netdepth_fine'], W=cfg_train['netwidth_fine'],
            input_ch=input_ch, output_ch=output_ch, skips=[4, ],
            input_ch_views=input_ch_views, use_viewdirs=cfg['rendering']['use_viewdirs'])

    return model, model_fine, embed_fn, embeddirs_fn

