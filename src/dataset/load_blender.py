import os
import numpy as np
import json
import cv2 as cv
import jittor as jt


trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(cv.imread(fname, cv.IMREAD_UNCHANGED))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [list(np.arange(counts[i], counts[i+1])) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        imgs = imgs_half_res

    # Cast intrinsics to right types
    H, W = int(H), int(W)

    loaded_data = {
        'imgs': imgs, 
        'poses': poses, 
        'render_poses': render_poses, 
        'calib': [H, W, focal], 
        'i_split': i_split}
    return loaded_data


def load_blender_data_ex(args):
    assert 'datadir' in args
    load_from_cache = ('load_cache' not in args) or (not args['load_cache'])
    if os.path.exists(os.path.join(args['datadir'], 'cache.bin')) and load_from_cache:
        print("Load from cache.")
        raw_data =  jt.load(os.path.join(args['datadir'], 'cache.bin'))
        for k in raw_data.keys():
            if isinstance(raw_data[k], np.ndarray):
                raw_data[k] = jt.array(raw_data[k])
        return raw_data

    load_arg = {'basedir': args['datadir']}
    if 'half_res' in args:
        load_arg['half_res'] = args['half_res']
    if 'testskip' in args:
        load_arg['testskip'] = args['testskip']

    # load blender data
    raw_data = load_blender_data(**load_arg)
    imgs = raw_data['imgs']
    if 'white_bkgd' in args and args['white_bkgd']:
        imgs = imgs[...,:3] * imgs[...,-1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[...,:3]
    raw_data['imgs'] = imgs

    i_train, i_val, i_test = raw_data['i_split']
    if 'fewshot' in args and args['fewshot'] > 0:
        if 'train_scene' in args:
            i_train = args['train_scene']
        else:
            np.random.seed(args['fewshot_seed'])
            i_train = np.random.choice(i_train, args['fewshot'], replace=False)
        print('Few-shot training ', i_train)
        raw_data['i_split'] = [i_train, i_val, i_test]
    
    # convert to jittor.Var
    for k in raw_data.keys():
        if isinstance(raw_data[k], np.ndarray):
            raw_data[k] = jt.array(raw_data[k])
    raw_data['imgs'] = raw_data['imgs'].permute(0, 3, 1, 2)
    jt.save(raw_data, os.path.join(args['datadir'], 'cache.bin'))
    print("Data cached.")
    return raw_data


if __name__ == '__main__':
    data = load_blender_data('data/nerf_synthetic/lego')
    print(data.keys())
