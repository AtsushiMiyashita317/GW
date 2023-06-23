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
        signal = F.pad(signal,(pad,pad))
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
