import torch

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

def interpolate(x:torch.Tensor, ilens:torch.Tensor, olens:torch.Tensor):
    """interpolate input batchwise

    Args:
        x (torch.Tensor): (B, T_i, F)
        ilens (torch.Tensor): (B,)
        olens (torch.Tensor): (B,)
    
    Returns:
        torch.Tensor: (B, T_o, F)
    """
    x = torch.nn.functional.pad(x, (0,0,0,2))
    max_olen = olens.max().item()
    # (B, 1)
    scales = ((ilens-1)/(olens-1))[:,None]
    # (B, 1)
    batch = torch.arange(x.size(0),device=x.device)[:,None]
    # (1, T_o)
    otime = torch.arange(max_olen,device=x.device)[None,:]
    # (B, T_o)
    idx = scales.mul(otime)
    idx[otime>=olens[:,None]] = -2
    # (B, T_o)
    idx0 = idx.floor().long()
    # (B, T_o)
    idx1 = idx0 + 1
    # (B, T_o, 1)
    t = (idx - idx0)[:,:,None]
    # (B, T_o, F)
    y = (1-t)*x[batch,idx0] + t*x[batch,idx1]
    return y