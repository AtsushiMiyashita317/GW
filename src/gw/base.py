import torch

class GW:
    @staticmethod
    def algebra(w:torch.Tensor) -> torch.Tensor:
        dim = w.shape[-1]
        w_ = torch.nn.functional.pad(w,[(dim-1)//2,(dim-1)//2])
        return torch.as_strided(
            w_,
            size=w_.shape[:-1]+(dim,dim),
            stride=w_.stride()[:-1]+(w_.stride(-1),w_.stride(-1))
        ).flip([-1]).mul(
            -1j*torch.arange(-dim//2+1,dim//2+1,device=w.device)
        )
    
    @staticmethod
    def element(w: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(GW.algebra(w))
    
    @staticmethod
    def signal_to_spectrum(signal:torch.Tensor) -> torch.Tensor:
        n = signal.size(-1)//2
        return torch.fft.fft(signal,dim=-1)[...,torch.arange(-n,n+1,device=signal.device)]
    
    @staticmethod
    def transform_to_map(transform:torch.Tensor) -> torch.Tensor:
        n = transform.size(-1)//2
        return torch.fft.ifft(
            torch.fft.fft(
                transform[...,:-1,:-1].roll((n,n),(-2,-1)),
                dim=-1
            ),
            dim=-2
        ).real
    
    @staticmethod
    def map(signal:torch.Tensor) -> torch.Tensor:
        """Create GW map from GW signal

        Args:
            signal (torch.Tensor): GW signal

        Returns:
            torch.Tensor: GW map
            
        Notes:
            signal length must be even
        """
        return GW.transform_to_map(GW.element(GW.signal_to_spectrum(signal)))
