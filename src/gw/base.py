import torch

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

def cubic_derivative(s:torch.Tensor, x:torch.Tensor, a:float=-0.5):
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
    h1 = -2*(a+3)*d1 + 3*(a+2)*d1**2
    d2 = torch.abs(xf.unsqueeze(-1) + torch.tensor([1,-2], device=s.device))
    h2 = 8*a - 10*a*d2 + 3*a*d2**2
    # (b,n,4)
    h = torch.stack([h2[:,:,0],h1[:,:,0],-h1[:,:,1],-h2[:,:,1]], dim=-1)
    # (b,1,1,1)
    b = torch.arange(s.size(0), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # (1,c,1,1)
    c = torch.arange(s.size(1), device=s.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    # (b,1,n,4)
    xi = xi.unsqueeze(1)
    s = torch.cat([2*s[...,:1]-s[...,1:2],s,2*s[...,-1:]-s[...,-5:-1].flip([-1])], dim=-1)
    return torch.einsum("icjk,ijk->icj", s[b,c,xi], h)

def gw(s:torch.Tensor, f:torch.Tensor=None, m:int=4):
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
