import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# important 이하의 전개가 모두 homogeneous coordinate로 전개한다.


# 아래의 lambda 식들은 novel view synthesis를 하기 위해서 camera view point를 y와 z(camera depth = radius of sphere)를 고정,
# x축을 시계 방향으로 돌리고 있다.

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# x축 축으로 yz의 회전
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# y축 축으로 xz의 회전
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

# z축을 기준으로 180도 돌림, conventional하게 보는 왼쪽 위에를 0,0으로 하기 위함
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)  # z 좌표를 radius만큼 뒤로 옮기기
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w  # 이부분이 Rz
    return c2w

# todo 여기서부터 분석 시작 그나마 자료구조가 간단하다.
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
            # json file에 대한 처리
            # camera extrinsic matrix

    all_imgs = []
    all_poses = []
    counts = [0]  # 뭘 세는 거지
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # test skip이 무엇인가?
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        #  전체 image들을 위해서 append 하여서 [number of image, 4] dimension을 가진 array의 생성, normalize 또한 시행
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])  # 누산 형으로 한다.
        all_imgs.append(imgs)  # train, val, test를 모두 합친 것
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    # [0, train_split), [train_split, train_split + val_split), [train_split + val_split, train_split + val_split + test_split)
    
    imgs = np.concatenate(all_imgs, 0)
    # axis = 0로서 하나의 통 array를 만든다. 기존에는 list의 느낌
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    # 어차피 전체 imgs의 H, W의 크기는 동일하다.
    camera_angle_x = float(meta['camera_angle_x'])
    # 화각을 의미하는 것 같다. zx plane에서 z 축을 중심으로 얼마나 x 축으로 벌어져 있는가

    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # focal length가 왜 이렇게 정의 되는 건가, 그림으로 그려보면 이해 된다.
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


