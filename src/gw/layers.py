import torch

from gw.gw import gw,gw2
import mytorch
import torchgroup


class GW(torchgroup.lie.LieGroup):
    def __init__(self, device=None) -> None:
        super().__init__()
        self._set(self._create_impl(device=device))
        
    def coef_al(self, dim:int, device=None):
        return -1j*(torch.abs(torch.arange(dim,device=device)[:,None]-torch.arange(dim,device=device)[None,:])<=(dim-1)//2)*\
            torch.arange(-dim//2+1,dim//2+1,device=device)
    
    def index_al(self, dim:int, device=None):
        i = torch.nn.functional.pad(
                torch.arange(
                    dim,
                    device=device,
                    dtype=torch.long   
                ),
                [(dim-1)//2,(dim-1)//2]
            )
        return torch.as_strided(
            i,
            size=(dim,dim),
            stride=(i.stride(-1),i.stride(-1))
        ).flip([-1]).flatten()
            
    def algebra(self, w:torch.Tensor):
        dim = w.shape[-1]
        w_ = torch.nn.functional.pad(w,[(dim-1)//2,(dim-1)//2])
        return torch.as_strided(
            w_,
            size=w_.shape[:-1]+(dim,dim),
            stride=w_.stride()[:-1]+(w_.stride(-1),w_.stride(-1))
        ).flip([-1]).mul(
            -1j*torch.arange(-dim//2+1,dim//2+1,device=w.device)
        )
        
    def adjoint(self, w:torch.Tensor):
        dim = w.shape[-1]
        w_ = torch.nn.functional.pad(w,[(dim-1)//2,(dim-1)//2])
        c = torch.arange(-3*(dim-1)//2,3*(dim-1)//2+1,device=w.device)
        return torch.as_strided(
                w_,
                size=w_.shape[:-1]+(dim,dim),
                stride=w_.stride()[:-1]+(w_.stride(-1),w_.stride(-1))
            ).flip([-1])*\
            torch.as_strided(
                -1j*c,
                size=(dim,dim),
                stride=(c.stride(-1),2*c.stride(-1))
            ).flip([-2])
    
    def derivative(self, y:torch.Tensor, dy:torch.Tensor, w:torch.Tensor):
        return torch.zeros(
            y.size()[:-2]+(y.size(-1),), 
            dtype=y.dtype, 
            device=y.device
        ).index_add_(
            -1,
            self.index_al(y.size(-1), device=y.device),
            ((y.transpose(-2,-1)@dy)*self.coef_al(y.size(-1), device=y.device)).flatten(-2)
        )
        
    def signal_to_spectrum(self, signal:torch.Tensor) -> torch.Tensor:
        n = signal.size(-1)//2
        return torch.fft.fft(signal,dim=-1)[...,torch.arange(-n,n+1,device=signal.device)]
    
    def transform_to_map(self, transform:torch.Tensor) -> torch.Tensor:
        n = transform.size(-1)//2
        return torch.fft.ifft(
            torch.fft.fft(
                transform[...,:-1,:-1].roll((n,n),(-2,-1)),
                dim=-1
            ),
            dim=-2
        ).real
    
    def map(self, signal:torch.Tensor) -> torch.Tensor:
        """Create GW map from GW signal

        Args:
            signal (torch.Tensor): GW signal

        Returns:
            torch.Tensor: GW map
            
        Notes:
            signal length must be even
        """
        return self.transform_to_map(self(self.signal_to_spectrum(signal)))
    
    def warp(self, signal:torch.Tensor, input:torch.Tensor) -> torch.Tensor:
        return self.map(signal).matmul(input.unsqueeze(-1)).squeeze(-1)
    

class SparseGW(GW):
    def __init__(self, n_neighbor, device=None) -> None:
        super().__init__(device)
        self.n_neighbor = n_neighbor

    @staticmethod
    def pulse_ifft(z:torch.Tensor,N:int,eps=1e-6):
        r = (1-z)*N
        return (1-z**N-r).div(r+eps).real-1/(2*N)+1
    
    @staticmethod
    def pulse_ifft_hann(z:torch.Tensor,N:int,eps=1e-6):
        h = SparseGW.pulse_ifft(z[...,None]*torch.arange(-1,2,device=z.device).mul(1j*torch.pi/N).exp(),N,eps)
        return h[...,0]/4+h[...,1]/2+h[...,2]/4

    @staticmethod
    def sparse_warp(z:torch.Tensor,N:int,n_neighbor:int):
        idx = z.angle().div(2*torch.pi/N,rounding_mode='floor')
        idx = idx[...,None]+torch.arange(-n_neighbor,n_neighbor+1,device=z.device)
        idx = idx.remainder(N)
        
        values = SparseGW.pulse_ifft_hann(z[...,None]*idx.mul(-2j*torch.pi/N).exp(),N//2).flatten()
        
        indices = mytorch.meshgrid(idx.size(),device=z.device)
        indices[-1] = idx
        indices = indices.flatten(start_dim=1)
        
        return torch.sparse_coo_tensor(indices=indices,values=values,size=list(z.shape)+[N])
    
    def transform_to_map(self, transform: torch.Tensor) -> torch.Tensor:
        n = transform.size(-1)
        z = mytorch.normalize(
                torch.fft.fft(
                    transform[...,:-1,(n+1)//2].roll(n//2,-1),
                    dim=-1
                )
            ).flip([-1])
        A = SparseGW.sparse_warp(z,n-1,self.n_neighbor)
        return A
    
    def warp(self, signal: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return mytorch.sparse.mv(self.map(signal),input)


class GW2D(torchgroup.lie.LieGroup):
    def __init__(self, device=None) -> None:
        super().__init__()
        self._set(self._create_impl(device=device))
        
    def __arange(self, size:torch.Size, device=None):
        x = torch.arange(-size[0]//2+1,size[0]//2+1,device=device)
        y = torch.arange(-size[1]//2+1,size[1]//2+1,device=device)
        return x[:,None], y[None,:]
    
    def __pad(self, w:torch.Tensor, scale:int, value=None):
        px = (w.size(-2)//2)*(scale-1)
        py = (w.size(-1)//2)*(scale-1)
        return torch.nn.functional.pad(w,[py,py,px,px],value=value)
    
    def __as_strided(self, w:torch.Tensor):
        size = (w.size(-2)//2+1,w.size(-1)//2+1)
        return torch.as_strided(
            w,
            size=w.size()[:-2]+size+size,
            stride=w.stride()[:-2]+w.stride()[-2:]+w.stride()[-2:]
        ).flip([-2,-1])
    
    def __index(self, size:torch.Size, device=None):
        return self.__as_strided(
            self.__pad(
                torch.arange(
                    size[0]*size[1],
                    device=device,
                    dtype=torch.long   
                ).unflatten(-1,size),
                2,
                value=size[0]*size[1]
            )
        ).flatten()
            
    def algebra(self, w:torch.Tensor):
        L = self.__as_strided(self.__pad(w,2))
        x,y = self.__arange(L.size()[-2:],device=w.device)
        return -1j*(L[...,0,:,:,:,:]*x+L[...,1,:,:,:,:]*y).flatten(-4,-3).flatten(-2,-1)
        
    def adjoint(self, w:torch.Tensor):
        L = self.__as_strided(self.__pad(w,2))
        A = torch.zeros(w.size()[:-3]+w.size()[-3:]+w.size()[-3:],device=w.device,dtype=w.dtype)
        x,y = self.__arange(L.size()[-2:],device=w.device)
        i0 = x[:,:,None,None]
        i1 = y[:,:,None,None]
        j0 = x[None,None,:,:]
        j1 = y[None,None,:,:]
        A[...,0,:,:,0,:,:] = (2*j0-i0)*L[0]+j1*L[1]
        A[...,0,:,:,1,:,:] = (j1-i1)*L[0]
        A[...,1,:,:,0,:,:] = (j0-i0)*L[1]
        A[...,1,:,:,1,:,:] = (2*j1-i1)*L[1]+j0*L[0]
        return -1j*A.flatten(-6,-4).flatten(-3,-1)
    
    def derivative(self, y:torch.Tensor, dy:torch.Tensor, w:torch.Tensor):
        cx,cy = self.__arange(w.size()[-2:],device=w.device)
        index = self.__index(w.size()[-2:],device=w.device)
        value = (y.transpose(-2,-1)@dy).unflatten(-2,w.size()[-2:]).unflatten(-1,w.size()[-2:])
        dw = torch.zeros_like(w).flatten(-2)
        zero = torch.zeros(dw.size()[:-2]+(dw.size(-1)+1,),device=dw.device,dtype=dw.dtype)
        dw[...,0,:] = zero.index_add(
            -1,
            index,
            value.mul(-1j*cx).flatten(-4)
        )[...,:-1]
        dw[...,1,:] = zero.index_add(
            -1,
            index,
            value.mul(-1j*cy).flatten(-4)
        )[...,:-1]
        return dw.flatten(-2)
    
    def derivative_(self,y,dy,w):
        al = self.algebra(torch.eye(w.size(-3)*w.size(-2)*w.size(-1),device=w.device,dtype=w.dtype).unflatten(-1,w.size()[-3:]))
        return torch.einsum('...ij,...ik,mkj->...m',dy,y,al)
        
    # def reg(self, dw:torch.Tensor, w:torch.Tensor) -> torch.Tensor:
    #     return dw + dw.abs().mean()*w*torch.arange(-w.size(-1)//2+1,w.size(-1)//2+1,device=w.device).abs()
    
    def signal_to_spectrum(self, signal:torch.Tensor) -> torch.Tensor:
        x,y = self.__arange((signal.size(-2)+1,signal.size(-1)+1))
        return torch.fft.fft2(signal,dim=[-2,-1])[...,x,y]
    
    def transform_to_map(self, transform:torch.Tensor, size:torch.Tensor) -> torch.Tensor:
        n = size[-2]//2,size[-1]//2
        return torch.fft.ifft2(
            torch.fft.fft2(
                transform.unflatten(-2,size).unflatten(-1,size)[...,:-1,:-1,:-1,:-1].roll((n[0],n[1],n[0],n[1]),(-4,-3,-2,-1)),
                dim=[-2,-1]
            ),
            dim=[-4,-3]
        ).real
    
    def map(self, signal:torch.Tensor) -> torch.Tensor:
        """Create GW map from GW signal

        Args:
            signal (torch.Tensor): GW signal

        Returns:
            torch.Tensor: GW map
            
        Notes:
            signal length must be even
        """
        return self.transform_to_map(self(self.signal_to_spectrum(signal)),(signal.size(-2)+1,signal.size(-1)+1))

    def warp(self, signal:torch.Tensor, input:torch.Tensor) -> torch.Tensor:
        map = self.map(signal)
        return map.flatten(-4,-3).flatten(-2,-1).matmul(input.flatten(-2,-1).unsqueeze(-1)).squeeze(-1).unflatten(-1,input.size()[-2:])


class STLinear(mytorch.autograd.Function):
    def __init__(self, window_size) -> None:
        super().__init__()
        
        class impl(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, w_inv=None):
                pass
            
            @staticmethod
            def backward(ctx, *grad_outputs):
                return super().backward(ctx, *grad_outputs)
            
        self.__func_impl = impl.apply()
    



class GW_Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


class GW_Conv1D(torch.nn.Module):
    def __init__(self, gw_signal_estimator, conv_layer, window_size, n) -> None:
        super().__init__()
        self.gw_signal_estimator = gw_signal_estimator
        self.conv_layer = conv_layer
        self.window_size = window_size
        self.n = n
        
    def forward(self, signal:torch.Tensor):
        gw_signal = self.gw_signal_estimator(signal)
        gw_signal,signal,length = gw.padding_for_stgw(gw_signal,signal,self.window_size)
        signal = gw.short_time_warp(gw_signal,signal,self.window_size,self.n)
        signal = self.conv_layer(signal)
        signal = gw.inverse_short_time_warp(gw_signal,signal,self.window_size,self.n)
        signal = gw.strip_for_stgw(signal,length,self.window_size)
        return signal

class GW_Conv2D(torch.nn.Module):
    def __init__(self, gw_signal_estimator, conv_layer, window_size, n) -> None:
        super().__init__()
        self.gw_signal_estimator = gw_signal_estimator
        self.conv_layer = conv_layer
        self.window_size = window_size
        self.n = n
        
    def forward(self, signal:torch.Tensor):
        gw_signal = self.gw_signal_estimator(signal)
        gw_signal,signal,length = gw2.padding_for_stgw(gw_signal,signal,self.window_size)
        signal = gw2.short_time_warp(gw_signal,signal,self.window_size,self.n)
        signal = self.conv_layer(signal)
        signal = gw2.inverse_short_time_warp(gw_signal,signal,self.window_size,self.n)
        signal = gw2.strip_for_stgw(signal,length,self.window_size)
        return signal

