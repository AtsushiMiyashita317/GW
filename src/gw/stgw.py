import torch
import gw.base as base

class stgw_impl(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx:torch.autograd.function.FunctionCtx, 
            x:torch.Tensor, 
            map:torch.Tensor, 
            map_inv:torch.Tensor,
            n_iter:int
        ):
        """stgw forward

        Args:
            ctx (torch.autograd.function.FunctionCtx)
            x (torch.Tensor, size=(batch, time, n_feature))
            map (torch.Tensor, size=(batch, 2, frame, window, window))
            map_inv (torch.Tensor, size=(batch, 2, frame, window, window))
            n_iter (int):

        Returns:
            torch.Tensor, size=(batch, time, n_feature)
        """
        window_size = map.size(-1)
        n = window_size//2
        a = x.new_zeros(x.size())
        b = x.new_zeros(x.size())
        a[:] = x[:]
        a_strided = stride_for_stgw(a, window_size)
        b_strided = stride_for_stgw(b, window_size)
        
        for _ in range(n_iter):
            torch.matmul(map[...,0,:,:,:], a_strided[...,0,:,:,:], out=b_strided[...,0,:,:,:])
            b[...,-n:,:] = a[...,-n:,:]
            torch.matmul(map[...,1,:,:,:], b_strided[...,1,:,:,:], out=a_strided[...,1,:,:,:])
            a[...,:n,:] = b[...,:n,:]
        y = a
        ctx.save_for_backward(y, map, map_inv)
        ctx.n_iter = n_iter
        return y
    
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx, dy:torch.Tensor):
        y:torch.Tensor
        map:torch.Tensor
        map_inv:torch.Tensor
        y, map, map_inv = ctx.saved_tensors
        n_iter = ctx.n_iter
        dmap = map_inv.new_zeros(map_inv.size())
        
        window_size = map.size(-1)
        n = window_size//2
        
        a = y.new_zeros(y.size())
        b = y.new_zeros(y.size())
        a[:] = y[:]
        a_strided = stride_for_stgw(a, window_size)
        b_strided = stride_for_stgw(b, window_size)
        
        da = dy.new_zeros(dy.size())
        db = dy.new_zeros(dy.size())
        da[:] = dy[:]
        da_strided = stride_for_stgw(da, window_size, transpose=True)
        db_strided = stride_for_stgw(db, window_size, transpose=True)
        
        for _ in range(n_iter):
            torch.matmul(map_inv[...,1,:,:,:], a_strided[...,1,:,:,:], out=b_strided[...,1,:,:,:])
            b[...,:n,:] = a[...,:n,:]
            dmap[...,1,:,:,:] += torch.matmul(b_strided[...,1,:,:,:], da_strided[...,1,:,:,:])
            torch.matmul(da_strided[...,1,:,:,:], map[...,1,:,:,:], out=db_strided[...,1,:,:,:])
            db[...,:n,:] = da[...,:n,:]
            torch.matmul(map_inv[...,0,:,:,:], b_strided[...,0,:,:,:], out=a_strided[...,0,:,:,:])
            a[...,-n:] = b[...,-n:]
            dmap[...,0,:,:,:] += torch.matmul(a_strided[...,0,:,:,:], db_strided[...,0,:,:,:])
            torch.matmul(db_strided[...,0,:,:,:], map[...,0,:,:,:], out=da_strided[...,0,:,:,:])
            da[...,-n:,:] = db[...,-n:,:]
        dx = da
        return dx, dmap.mT, None, None
    
def padding_for_stgw(gw_signal:torch.Tensor,signal:torch.Tensor,window_size:int):
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
    hop_length = window_size//2
    n_window = (length+hop_length-1)//hop_length
    n_window = n_window//2+1
    whole_length = window_size*n_window+hop_length
    # padding
    pad = hop_length,whole_length-length-hop_length
    signal = torch.nn.functional.pad(signal,(0,0)+pad)
    gw_signal = torch.nn.functional.pad(gw_signal,pad)
    return gw_signal,signal,length

def strip_for_stgw(signal:torch.Tensor,length:int,window_size:int):
    return signal[...,window_size//2:window_size//2+length,:]

def stride_for_stgw(x:torch.Tensor, window_size:int, dim=-2, transpose=False) -> torch.Tensor:
    dim = (x.ndim+dim)%x.ndim
    if transpose:
        size = x.size()[:dim] + (2, x.size(dim)//window_size) + x.size()[dim+1:] + (window_size,)
        stride = x.stride()[:dim] + (window_size//2*x.stride(dim), window_size*x.stride(dim)) + x.stride()[dim+1:] + (x.stride(dim),)
    else:
        size = x.size()[:dim] + (2, x.size(dim)//window_size, window_size) + x.size()[dim+1:]
        stride = x.stride()[:dim] + (window_size//2*x.stride(dim), window_size*x.stride(dim), x.stride(dim)) + x.stride()[dim+1:]
    return x.as_strided(size, stride)

def stgw(gw_signal:torch.Tensor, input:torch.Tensor, window_size:int, n_iter:int) -> torch.Tensor:
    w,x,length = padding_for_stgw(gw_signal, input, window_size)
    w = stride_for_stgw(w, window_size, dim=-1)*torch.hann_window(window_size, dtype=w.dtype, device=w.device)
    w = w/n_iter
    
    map = base.GW.map(w)
    map_inv = base.GW.map(-w)
    
    x = stgw_impl.apply(x, map, map_inv, n_iter)
    
    output = strip_for_stgw(x, length, window_size)
    
    return output
