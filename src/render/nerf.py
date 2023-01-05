import jittor as jt
import jittor.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def run_network(inputs, viewdirs, model, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Run NeRF model (embedding + MLP)
    inputs: [..., 3]
    viewdirs: [..., 3]
    """
    use_dir = (viewdirs is not None) and (embeddirs_fn is not None)
    input_shape = inputs.shape
    point_num = 1
    for _d in range(len(input_shape) - 1): point_num *= input_shape[_d]
    group_size = netchunk
    group_num = (
        (point_num // group_size) if (point_num % group_size == 0) else (point_num // group_size + 1))
    if group_num == 0:
        group_num = 1
        group_size = point_num

    pt_group_output = []
    inputs = inputs.reshape([-1, input_shape[-1]])  # [N, 3], flatten points
    if use_dir:
        viewdirs = viewdirs.reshape([-1, viewdirs.shape[-1]])  # [N, 3]
    for gi in range(group_num):
        start = gi * group_size
        end = (gi + 1) * group_size
        end = (end if (end <= point_num) else point_num)

        pt_group = inputs[start:end, :]  # [group_p_num, 3]
        embedded = embed_fn(pt_group)  # [group_p_num, ch]

        if use_dir:
            dir_group = viewdirs[start:end, :]  # [group_p_num, 3]
            embedded_dirs = embeddirs_fn(dir_group)  # [group_p_num, ch]
            embedded = jt.concat([embedded, embedded_dirs], dim=-1)
        pt_group_output.append(model(embedded))  # [group_p_num, out_ch]
    
    output_flat = jt.concat(pt_group_output, dim=0)  # [N, , out_ch]
    return jt.reshape(output_flat, list(input_shape[:-1]) + [output_flat.shape[-1], ])


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        NeRF MLP 
        Args:
            D: number of layers of MLP
            W: dimension (depth) of MLP
            input_ch: number of channels of input 
            input_ch_views: number of channels of direction map
            output_ch: output channels (usually rgb + sigma)
            skips: input skip-connection
            use_viewdirs: use view direction as an additional condition of RGB 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.Sequential(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.Sequential([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, x):
        if not self.use_viewdirs and x.shape[-1] == self.input_ch:
            input_pts = x
        else:
            input_pts = x[..., :self.input_ch]
            input_views = x[..., self.input_ch:]
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = nn.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = jt.concat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for _layer in self.views_linears:
                h = nn.relu(_layer(h))

            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs    


class NeRF_RGB(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, alpha_model=None):
        """ 
        """
        super(NeRF_RGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.Sequential(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.Sequential([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.alpha_model = alpha_model

    def execute(self, x):
        if not self.use_viewdirs and x.shape[-1] == self.input_ch:
            input_pts = x
        else:
            input_pts = x[..., :self.input_ch]
            input_views = x[..., self.input_ch:]
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = nn.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = jt.concat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            with jt.no_grad():
                alpha = self.alpha_model(x)[..., 3][..., None]
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], dim=-1)
        
            for _layer in self.views_linears:
                h = nn.relu(_layer(h))

            rgb = self.rgb_linear(h)
            outputs = nn.concat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs    


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    test_data = jt.rand(1024, 64, 6, dtype=jt.float32)
    model = NeRF(use_viewdirs=True)
    output = model(test_data)
    print(output.shape)
