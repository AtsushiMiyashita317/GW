import torch
import gw.base as base

class stgw_impl(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx:torch.autograd.function.FunctionCtx,
            n_iter:int,
            x:torch.Tensor, 
            map:torch.Tensor, 
            map_inv:torch.Tensor=None,
        ):
        """stgw forward

        Args:
            ctx (torch.autograd.function.FunctionCtx)
            n_iter (int):
            x (torch.Tensor, size=(batch, time, n_feature))
            map (torch.Tensor, size=(batch, overlap, frame, window, window))
            map_inv (torch.Tensor, size=(batch, orverlap, frame, window, window))

        Returns:
            torch.Tensor, size=(batch, time, n_feature)
        """
        window_size = map.size(-1)
        overlap = map.size(-4)
        a = x.new_zeros(x.size())
        a.copy_(x)
        a_strided = stride_for_stgw(a, window_size, overlap)
        
        for _ in range(n_iter):
            for i in range(overlap):
                a_strided[...,i,:,:,:] = torch.matmul(map[...,i,:,:,:], a_strided[...,i,:,:,:])
        y = a
        if map_inv is not None:
            ctx.save_for_backward(y, map, map_inv)
        else:
            ctx.save_for_backward(y, map)
        ctx.save_inv = map_inv is not None
        ctx.n_iter = n_iter
        return y
    
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx, dy:torch.Tensor):
        y:torch.Tensor
        map:torch.Tensor
        map_inv:torch.Tensor
        if ctx.save_inv:
            y, map, map_inv = ctx.saved_tensors
        else:
            y, map = ctx.saved_tensors
            map_inv = torch.linalg.inv(map)
        n_iter = ctx.n_iter
        dmap = map_inv.new_zeros(map_inv.size())
        
        window_size = map.size(-1)
        overlap = map.size(-4)
        
        a = y.new_zeros(y.size())
        a.copy_(y)
        a_strided = stride_for_stgw(a, window_size, overlap)
        
        da = dy.new_zeros(dy.size())
        da.copy_(dy)
        da_strided = stride_for_stgw(da, window_size, overlap,transpose=True)
        
        for _ in range(n_iter):
            for i in range(overlap):
                a_strided[...,-i-1,:,:,:] = torch.matmul(map_inv[...,-i-1,:,:,:], a_strided[...,-i-1,:,:,:])
                dmap[...,-i-1,:,:,:] += torch.matmul(a_strided[...,-i-1,:,:,:], da_strided[...,-i-1,:,:,:])
                da_strided[...,-i-1,:,:,:] = torch.matmul(da_strided[...,-i-1,:,:,:], map[...,-i-1,:,:,:])
        dx = da
        return None, dx, dmap.transpose(-2,-1), None
    
def padding_for_stgw(gw_signal:torch.Tensor,signal:torch.Tensor,window_size:int,overlap:int):
    """preprocess of stgw

    Args:
        gw_signal (torch.Tensor, size=(batch, time))
        signal (torch.Tensor, size=(batch, time, n_feature)) 
        window_size (int)

    Returns:
        (torch.Tensor, size=(batch, time)), (torch.Tensor, size=(batch, time, n_feature))
    """
    # calculate size
    length = gw_signal.size(-1)
    hop_length = window_size//overlap
    n_window = (length+hop_length-1)//hop_length
    n_window = n_window//overlap+1
    whole_length = window_size*(n_window+1)-hop_length
    # padding
    pad = window_size//2,whole_length-length-window_size//2
    signal = torch.nn.functional.pad(signal,(0,0)+pad)
    gw_signal = torch.nn.functional.pad(gw_signal,pad)
    return gw_signal,signal,length

def strip_for_stgw(signal:torch.Tensor,length:int,window_size:int):
    return signal[...,window_size//2:window_size//2+length,:]

def stride_for_stgw(x:torch.Tensor, window_size:int, overlap=2, dim=-2, transpose=False) -> torch.Tensor:
    dim = (x.ndim+dim)%x.ndim
    if transpose:
        size = x.size()[:dim] + (overlap, x.size(dim)//window_size) + x.size()[dim+1:] + (window_size,)
        stride = x.stride()[:dim] + (window_size//overlap*x.stride(dim), window_size*x.stride(dim)) + x.stride()[dim+1:] + (x.stride(dim),)
    else:
        size = x.size()[:dim] + (overlap, x.size(dim)//window_size, window_size) + x.size()[dim+1:]
        stride = x.stride()[:dim] + (window_size//overlap*x.stride(dim), window_size*x.stride(dim), x.stride(dim)) + x.stride()[dim+1:]
    return x.as_strided(size, stride)

class LPF(torch.nn.Module):
    def __init__(self, window_size:int) -> None:
        super().__init__()
        self.window_size = window_size
        filter = torch.fft.ifft(
            1/torch.linspace(
                -128,
                128,
                window_size+1
            ).roll(
                window_size//2,
                0
            ).square().add(1)
        ).real.roll(
            window_size//2,
            0
        )
        self.register_buffer('filter', filter)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x,(self.window_size//2,self.window_size//2))
        return x.as_strided(
            x.size()[:-1] + (x.size(-1)-self.window_size, self.window_size+1),
            x.stride()[:-1] + (x.stride(-1), x.stride(-1))
        ).matmul(self.filter)

def stgw(gw_signal:torch.Tensor, input:torch.Tensor, window_size:int, overlap:int, n_iter:int, scale=2, pad=0) -> torch.Tensor:
    w,x,length = padding_for_stgw(gw_signal, input, window_size, overlap)
    w = stride_for_stgw(w, window_size, overlap, dim=-1)*torch.hann_window(window_size, dtype=w.dtype, device=w.device)
    w = w/overlap/n_iter*2
    
    map = base.GW.map(w, exp_size=window_size*scale+1, pad=pad)
    
    x = stgw_impl.apply(n_iter, x, map)
    
    output = strip_for_stgw(x, length, window_size)
    
    return output
