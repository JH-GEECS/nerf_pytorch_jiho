import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # 이부분은 살짝 이해가 안간다. 왜 raw input을 그대로 이용하는가?
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        """
        어려웠는데 torch.linspace(start, end, steps)를 몰라서 그러하다.
        (start, start + (end - start)/(steps - 1),..., start + (steps - 2) * (end - start)/ (steps - 1), end)
        [0, 1, 2, ..., step - 1]
        """
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            # [2^0, 2^1,..., 2^9]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            # [2^0, 2^0 + (2^9 - 2^0)/(10-1),..., 2^0 + (2^9 - 2^0)]
            
        # frequency를 L개로, 그리고 sin, cos 2개 이용 => 2*L*d 개의 dimension 증가
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# positional encoding 생성 부로 보이는 데 계속 살펴본다.
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    """
    *args는 tuple 형태로 넘기므로, array처럼 접근이 가능하다
    **kyargs(keyword arguments)는 dict 형태로 넘기므로, name을 통해서 접근이 가능하다.
    """
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        NeRF의 neural network는 단순한 MLP이다.


        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # 중간의 output 뽑아내기 전까지의 hidden layer
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # 디버깅을 하는 것이 아니라면, non-Lambertian reflectance을 고려하기 위해서는 필요하다.
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # alpha 추출 layer와 같은 depth에서 color 추출을 위해서 하나의 hidden layer 추가

            self.alpha_linear = nn.Linear(W, 1)
            # MLP를 통해서 hidden layer 이후에 alpha 추출

            self.rgb_linear = nn.Linear(W//2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        """
        torch.split(tensor, split_size_or_sections, dim=0)
            tensor(tensor)
                tensor to split
            split_size_or_sections(int) or (list(int))
                size of a single chunk or list of sizes for each chunk
            dim(int) -> 여기서는 batch x input 이므로 input을 split하기 위하여 위와 같은 함수를 사용하였다.
                dimension along which to split the tensor
        """
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # todo batch norm을 사용하지 않는 이유는 무엇인가?
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 여기는 무엇인가?
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            # self.alpha_linear = nn.Linear(W, 1)
            feature = self.feature_linear(h)
            # self.feature_linear = nn.Linear(W, W)
            h = torch.cat([feature, input_views], -1)
            # 단순히 hidden layer의 feature와 view direction을 고려한 것이다.
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
        # [r, g, b, d]로 된다.

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
"""

여기는 무엇을 하는 곳인가? 어렵다.
ray를 조사해야 하는데 그를 위한 기준점을 만드는 것을 수행한다.

"""
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    """
    torch.meshgrid은 무엇인가? -> https://seong6496.tistory.com/129와 비슷한 느낌이다.
    뭔가 격자를 만드는 느낌
    
    해당 함수에서는 이를 pixel 단위로 격자를 만들기 위하여 사용하였다.
    
    x,y coordinate와 i,j indexing 간에는 약간의 차이가 있다.
        [y][x] - cartesian, [i][j] - matrix indexing
        
    important
        매우 중요한데 image에서의 시작은 좌측 상단에서 시작하고 오른쪽이 +x, 아래쪽이 +y
    
    """

    i = i.t()  # x에 해당
    j = j.t()  # y에 해당
    # matrix indexing에서 cartesian coordinate로의 전환을 위해서 transpose 한다.
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # normalized camera frame과 camera frame간의 치환을 위한 식으로 보인다, intrinsic matrix 전환 꼴이 보인다.
    # z 축이 -1인 것과 y축에 -1이 곱해진 것이 이해가 안가는데, -> open GL rendering에 따른 것인데 여기 부분 살펴보기
    # x축을 기준으로 180도 회전시켰다고 가정하면 이해가 된다, ray가 camera 원점으로 들어오는 것으로 하기 위함이다.
    # blender_cv_translation.py를 보면 53 ~ 62번 줄에서 설명이 되어있다. blender의 좌표계와
    # computer vision 좌표계랑 맞추위 위한 연산 이다.
    # K = blender camera intrinsic matrix, R = blender camera rotation matrix, 즉 pixel 좌표계를 blender camera 좌표계로 옮김

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 하... 어려운데 직접 써보니 된다. 1. broadcasting이랑 2. matrix multiplication인데 자세한 건 onenote NeRF 코드 분석 참조
    # 굳이 einsum을 사용할 필요는 없을 듯 하다. 선형대수학을 공부해야한다.

    """
    np.newaxis = None
    c2w가 별것이 아니다. E matrix에서 R부분만 떼어낸 것이다
    """
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 논문에서의 의문이 드디여 해소, 모든 ray의 원점은 world coordinate에서의 camera의 좌표이다.
    # 또한 blender에서의 matrix_world는 camera의 world coordinate를 의미한다.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    # bins가 z_vals_mid 이다.
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 여기에서 normalized를 시행하여 더한다.
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
    # [0, p(1), p(1) + p(2), ... ,p(1)+...+p(n-1),1]의 구조
    # cdf라서 무조건 0과 1에 bound 된다.

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        # u를 일정한 간격으로 쪼개기
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        # [1024, 128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # categorical method ,alias method
    # finite discrete distribution에 대하여 non-uniform random variate generation 방법론이 있는 것 같다.

    # Invert CDF
    u = u.contiguous()
    # 여기는 왜 contiguous로 선언한 것일까? https://jimmy-ai.tistory.com/122
    # view(), expand(), transpose() 등의 method를 사용할 경우에
    # index의 순서와 memory 배열을 일치시키고자 한다.
    inds = torch.searchsorted(cdf, u, right=True)
    # [cdf[0], cdf[1]), [cdf[1], cdf[2]) ... [cdf[n-1], cdf[n]) 에서 u가 어디에 속하는 지를 구한다.
    # 0, 1, 2, 3, ..., 만일 값이 [cdf[0], cdf[1]) 사이 라면 1의 index를 반환한다.
    # 이부분이 F^-1(X)에 해당함, 어떤 구간에 속하는 지를 찾아내는 것임
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    # 0 이상이 되도록, 그리하여 위에서 1의 index를 가진다면 0을 반환하도록
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    # cdf[n-1]의 n-1 이하가 되도록, 그리하여 위에서 1의 index를 가진다면 1을 반환하도록
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    # g가 gather의 의미인듯

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # [1024, 128, 63], 각 배치에 대하여 각 레이에 대하여, u가 어디에 속하는지?
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # expand가 dim value = 1 인 곳을 복사한다는 의미, 다만 memory efficiency를 위해서 값을 진짜 복사 하지는 않음
    # index가 아닌 [0,1]에서 실제 어느 범위에 속하는 지를 return 반환하기 위함
    # [1024, 128, 63, 2]의 느낌
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 이건 z values 값 사이이다.

    denom = (cdf_g[...,1]-cdf_g[...,0])
    # above와 below의 차이 P(X[i-1] <= x < X[i])의 느낌
    # denominator?
    # 기존의 coarse sampling 시의 간격에 대한 값들이다.
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    # cdf 사이 값이 0에 가깝다면(pdf = 0) 1을 부여하고 아니라면, pdf 값을 부여한다.
    # 32비트 float의 경우에는 유효숫자가 6자리이다.
    """
    torch.where(condition, x, y)
    condition = true 이면 x, 아니면 y
    
    뭔가 차이가 적다면(즉 해당 구간에 분포가 적다면) 다음으로 넘기는 느낌, 아니면 해당 구간에서 놀도록 하는 느낌
    
    """

    # Convert samples to ray length.
    t = (u-cdf_g[...,0])/denom
    # (u - cdf_g[...,0]는 기본적으로 u와 cdf 모두 128등분을 하였고 이 둘을 뺸다는 것은 그 사이의 작은 값, 다음 구간으로 넘어 가기 전의
    # 값을 얻겠다는 의미이다.
    # t를 0과 1사이의 값을 얻고자 한다.
    # 그런데 만일 cdf 사이의 값이 매우 작다면 1을 부여하곘다는 의미인데,
    # 즉 두 idx 사이에 거의 아무 것도 없다. 1/128보다 cdf 값이 적다
    # Q. 왜 하필 1e-5인것 인가 u - cdf[...,0]이 1e-5보다 작은 값을 얻을 수 없는 경우를 생각해본다.
    # A. 위에서 below, above와 같은 경우 라면 cdf_g에서 같은 idx를 가리킬 경우도 있다. 이러면 0인데
    # float의 특성을 생각한다면 == 연산자는 쓸 수 없다. 그렇기에 차이가 1e-5 이런식으로 한 것 같다.

    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    # 기존의 bing_g는 uniform한 부분에서 추출된 것, 즉, discrete하면서도 z의 분포를 나타낼 수 있는 것으로 보임
    # 하지만 inversed sampling을 거쳤으므로 많이 분포한 부분에 위치한 bins_g[...,0]가 많이 뽑힌다.
    #
    # 다만 t 부분 만일 cdf 차가 매우 작다면 해당 부분에 분포가 집중된다.

    return samples
    # https://en.wikipedia.org/wiki/Non-uniform_random_variate_generation 여기 공부하기
