import torch
import torch.nn.functional as F

import matrix_exp

class mexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return matrix_exp.forward(x, 2048)
    
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return matrix_exp.backward(x, grad, 2048)

class GW:
    @staticmethod
    def algebra(w:torch.Tensor, dim:int):
        p = 2*dim-1-w.size(-1)
        w_ = F.pad(w,[(p+1)//2,p//2])
        return torch.as_strided(
            w_,
            size=w_.shape[:-1]+(dim,dim),
            stride=w_.stride()[:-1]+(w_.stride(-1),w_.stride(-1))
        ).flip([-1]).mul(
            -1j*torch.arange(-dim//2+1,dim//2+1,device=w.device)
        )
        
    @staticmethod
    def element(w: torch.Tensor, exp_size:int) -> torch.Tensor:
        if exp_size is None:
            exp_size = w.size(-1)*2-1
        return mexp.apply(GW.algebra(w, exp_size))
    
    @staticmethod
    def spectrogram_to_signal(spectrogram: torch.Tensor):
        """spectrogram to signal
        Args:
            spectrogram (torch.Tensor): (...,T,F), normally distributed

        Returns:
            torch.Tensor: (...,T)
        """
        spectrogram = spectrogram.unsqueeze(0)
        f = spectrogram.size(-1)
        t = spectrogram.size(-2)
        b = spectrogram.size()[:-2]
        spectrogram = spectrogram.flatten(end_dim=-3)
        spectrogram[:,:,0] = spectrogram.real[:,:,0]
        spectrogram[:,:,1:] = spectrogram[:,:,1:].div(torch.arange(1,f,device=spectrogram.device))
        signal = torch.fft.irfft(spectrogram,dim=-1)
        signal = F.pad(signal, [0,1])
        signal = F.fold(signal,output_size=(1,t),kernel_size=(1,t),padding=(0,f-1))*2*f
        signal = signal[:,0,0,:]
        signal = signal.unflatten(0,b).squeeze(0)
        return signal

    @staticmethod        
    def signal_to_spectrum(signal:torch.Tensor, pad:int) -> torch.Tensor:
        p = torch.linspace(0,1,pad,device=signal.device)
        signal = torch.cat([p*signal[...,0:1],signal,p.flip([0])*signal[...,-2:-1]],dim=-1)
        # signal = F.pad(signal,(pad,pad))
        n = signal.size(-1)//2
        return torch.fft.fft(signal,dim=-1)[...,torch.arange(-n,n+1,device=signal.device)]
    
    @staticmethod
    def transform_to_map(transform:torch.Tensor, n:int) -> torch.Tensor:
        p = transform.size(-1)-n
        transform = transform[...,(p+1)//2:(p+1)//2+n,(p+1)//2:(p+1)//2+n]
        return torch.fft.ifft(
            torch.fft.fft(
                transform.roll((n//2+1,n//2+1),(-2,-1)),
                dim=-1
            ),
            dim=-2
        ).real
    
    @staticmethod
    def map(signal:torch.Tensor, exp_size:int=None, pad:int=0, sr=1) -> torch.Tensor:
        """Create GW map from GW signal

        Args:
            signal (torch.Tensor): GW signal

        Returns:
            torch.Tensor: GW map
            
        Notes:
            signal length must be even
        """
        return GW.transform_to_map(
            GW.element(
                GW.signal_to_spectrum(signal*(sr/(signal.size(-1)+2*pad))**2, pad), 
                exp_size
            ), 
            signal.size(-1)+2*pad
        )[...,pad:pad+signal.size(-1),pad:pad+signal.size(-1)]

def cubic_interpolation(s:torch.Tensor, x:torch.Tensor, a:float=-0.5):
    """Cubic interpolation

    Args:
        s (torch.Tensor, (b,c,n)): input signal
        x (torch.Tensor, (b,n)): coordinate
        a (float, optional): Interpolation parameter. Defaults to -0.5.

    Returns:
        torch.Tensor, (b,c,n): interpolated signal
    """
    # (b,n)
    x = torch.clamp(x, min=0, max=s.size(-1))
    xi = x.floor()
    xf = x - xi
    # (b,n,4)
    xi = xi.long().unsqueeze(-1) + torch.arange(4, device=s.device)
    # (b,n,2)
    d1 = torch.abs(xf.unsqueeze(-1) + torch.tensor([0,-1], device=s.device))
    h1 = 1 - (a+3)*d1**2 + (a+2)*d1**3
    d2 = torch.abs(xf.unsqueeze(-1) + torch.tensor([1,-2], device=s.device))
    h2 = -4*a + 8*a*d2 - 5*a*d2**2 + a*d2**3
    # (b,n,4)
    h = torch.stack([h2[:,:,0],h1[:,:,0],h1[:,:,1],h2[:,:,1]], dim=-1)
    # (b,1,1,1)
    b = torch.arange(s.size(0), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # (1,c,1,1)
    c = torch.arange(s.size(1), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    # (b,1,n,4)
    xi = xi.unsqueeze(1)
    s = torch.nn.functional.pad(s, [1,4], mode='replicate')
    return torch.einsum("icjk,ijk->icj", s[b,c,xi], h)

def gw_ode(s:torch.Tensor, f:torch.Tensor=None, m:int=4):
    """General

    Args:
        s (torch.Tensor, (b,n)): GW signal
        f (torch.Tensor, (b,n), optional): Initial warping function
        m (int, optional): Number of iteration. Defaults to 4.

    Returns:
        torch.Tensor, (b,n): warping function
    """
    # (b,n)
    if f is None:
        f = torch.arange(s.size(-1), device=s.device).expand_as(s)
    # (b,1,n)
    s = s.unsqueeze(1)
    for _ in range(m):
        # (b,n)
        k1 = cubic_interpolation(s, f).squeeze(1).div(m)
        k2 = cubic_interpolation(s, f+k1/2).squeeze(1).div(m)
        k3 = cubic_interpolation(s, f+k2/2).squeeze(1).div(m)
        k4 = cubic_interpolation(s, f+k3).squeeze(1).div(m)
        f = f + (k1+2*k2+2*k3+k4)/6
    return f

def bicubic_interpolation(s:torch.Tensor, z:torch.Tensor, a:float=-0.5):
    """Bicubic interpolation

    Args:
        s (torch.Tensor, (b,c,nx,ny)): input signal
        z (torch.Tensor, (b,2,nx,ny)): coordinate
        a (float, optional): interpolation parameter. Defaults to -0.5.

    Returns:
        torch.Tensor (b,nx,ny): interpolated signal
    """
    x = z[:,0]
    y = z[:,1]
    x = torch.clamp(x, min=0, max=s.size(-2))
    y = torch.clamp(y, min=0, max=s.size(-1))
    # (b,2,nx,ny)
    z = torch.stack([x,y],dim=1)
    zi = z.floor()
    zf = z - zi
    # (b,2,nx,ny,4)
    zi = zi.long().unsqueeze(-1) + torch.arange(4, device=s.device)
    # (b,2,nx,ny,2)
    d1 = torch.abs(zf.unsqueeze(-1) + torch.tensor([0,-1], device=s.device))
    h1 = 1 - (a+3)*d1**2 + (a+2)*d1**3
    d2 = torch.abs(zf.unsqueeze(-1) + torch.tensor([1,-2], device=s.device))
    h2 = -4*a + 8*a*d2 - 5*a*d2**2 + a*d2**3
    # (b,2,nx,ny,4)
    h = torch.stack([h2[...,0],h1[...,0],h1[...,1],h2[...,1]], dim=-1)
    # (b,nx,ny,4)
    hx = h[:,0]
    # (b,nx,ny,4)
    hy = h[:,1]
    # (b,1,nx,ny,4,1)
    xi = zi[:,0].unsqueeze(-1).unsqueeze(1)
    # (b,1,nx,ny,1,4)
    yi = zi[:,1].unsqueeze(-2).unsqueeze(1)
    # (b,1,1,1,1,1)
    b = torch.arange(s.size(0), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # (1,c,1,1,1,1)
    c = torch.arange(s.size(1), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    # (b,c,nx,ny)
    ps = torch.nn.functional.pad(s,(1,10,1,10), mode='replicate')
    # (b,c,nx,ny)
    return torch.einsum("icjklm,ijkl,ijkm->icjk", ps[b,c,xi,yi], hx, hy)

def gw2d(s:torch.Tensor, f:torch.Tensor=None, m:int=4):
    """General warping for 2d tensor

    Args:
        s (torch.Tensor, (b,2,nx,ny)): GW signal
        f (torch.Tensor, (b,2,nx,ny), optional): Initial warping function
        m (int, optional): number of iteration. Defaults to 4.

    Returns:
        torch.Tensor, (b2,,nx,ny): sampled warping function
    """
    if f is None:
        # (nx,1)
        fx = torch.arange(s.size(-2), device=s.device).unsqueeze(1)
        # (1,ny)
        fy = torch.arange(s.size(-1), device=s.device).unsqueeze(0)
        fx, fy = torch.broadcast_tensors(fx, fy)
        # (2,nx,ny) -> (b,2,nx,ny)
        f = torch.stack([fx,fy], dim=0).unsqueeze(0).expand_as(s)
    for _ in range(m):
        # (b,2,nx,ny)
        k1 = bicubic_interpolation(s, f).div(m)
        k2 = bicubic_interpolation(s, f+k1/2).div(m)
        k3 = bicubic_interpolation(s, f+k2/2).div(m)
        k4 = bicubic_interpolation(s, f+k3).div(m)
        f = f + (k1+2*k2+2*k3+k4)/6
    return f
