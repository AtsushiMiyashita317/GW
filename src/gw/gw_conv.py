import math

import torch


class GWConv1d(torch.nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        hidden_channels:int,
        kernel_size:int, 
        depth:int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kernel_size%2==1, 'kernel_size must be odd'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.kwargs = kwargs
        self.kwargs['padding'] = self.kernel_size // 2
        self.depth = depth

        self.emb = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = torch.nn.Conv1d(2*hidden_channels, out_channels, 1)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1)

        self.kernel = torch.nn.Parameter(
            torch.zeros(hidden_channels, hidden_channels, kernel_size, depth, dtype=torch.cfloat)
        )
        self._reset_parameters
    
    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform(self.kernel, a=math.sqrt(5), mode='fan_out')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        x = self.emb(x)
        x = torch.fft.fft(x, dim=-1)
        x = x.mul(1j*torch.fft.fftfreq(x.size(-1), device=x.device))
        x = torch.fft.fftshift(x, dim=-1)
        kernel = self.kernel + self.kernel.conj().flip(2)
        for i in range(self.depth):
            x = torch.nn.functional.conv1d(x, kernel[...,i], **self.kwargs)
        x = torch.fft.ifftshift(x, dim=-1)
        x = torch.fft.ifft(x, dim=-1)
        x = x.real
        # x = torch.layer_norm(x, x.size()[1:])
        x = self.proj(x)
        x = res + 1e-3 * x
        return x


class DynamicGWConv1d(torch.nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        hidden_channels:int,
        kernel_size:int, 
        depth:int,
        classes:int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kernel_size%2==1, 'kernel_size must be odd'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.kwargs = kwargs
        self.kwargs['padding'] = self.kernel_size // 2
        self.depth = depth
        self.classes = classes
        
        self.emb = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = torch.nn.Conv1d(2*hidden_channels, out_channels, 1)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 1)

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, depth*classes),
            torch.nn.Unflatten(-1, (depth, classes)),
            torch.nn.Softmax(dim=-1),
        )

        self.kernel = torch.nn.Parameter(
            torch.zeros(2*hidden_channels, hidden_channels, kernel_size, 1, depth, classes, dtype=torch.cfloat)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform(self.kernel, a=math.sqrt(5), mode='fan_out')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        x = self.emb(x)
        x = torch.fft.fft(x, dim=-1)
        attn = self.attn(x.real.select(-1, 0))
        x = x.mul(1j*torch.fft.fftfreq(x.size(-1), device=x.device))
        x = torch.fft.fftshift(x, dim=-1)
        size = x.size()[:2]
        x = x.flatten(0, 1).unsqueeze(0)
        kernel = self.kernel + self.kernel.conj().flip(2)
        kernel = self.kernel.mul(attn.cfloat()).sum(-1)
        kernel = kernel.permute(3, 0, 1, 2, 4).flatten(0, 1)
        for i in range(self.depth):
            x = torch.nn.functional.conv1d(x, kernel[...,i], group=size[0], **self.kwargs)
        x = x.squeeze(0).unflatten(0, size)
        x = torch.fft.ifftshift(x, dim=-1)
        x = torch.fft.ifft(x, dim=-1)
        x = x.real
        # x = torch.layer_norm(x, x.size()[1:])
        x = self.proj(x)
        x = res + 1e-3 * x
        return x


class GWConv2d(torch.nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        hidden_channels:int,
        kernel_size:int, 
        depth:int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kernel_size%2==1, 'kernel_size must be odd'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.kwargs = kwargs
        self.kwargs['padding'] = self.kernel_size // 2
        self.kwargs['groups'] = 2

        self.emb = torch.nn.Conv2d(in_channels, hidden_channels, 1)
        self.proj = torch.nn.Conv2d(2*hidden_channels, out_channels, 1)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)

        self.kernel = torch.nn.Parameter(
            torch.randn(2*hidden_channels, hidden_channels, kernel_size, kernel_size, depth, dtype=torch.cfloat)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform(self.kernel, a=math.sqrt(5), mode='fan_out')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        x = self.emb(x)
        x = torch.fft.fft2(x)
        x1 = x.mul(1j*torch.fft.fftfreq(x.size(-2), device=x.device).unsqueeze(-1))
        x2 = x.mul(1j*torch.fft.fftfreq(x.size(-1), device=x.device).unsqueeze(-2))
        x = torch.cat([x1, x2], dim=-3)
        x = torch.fft.fftshift(x, dim=(-2,-1))
        kernel = self.kernel + self.kernel.conj().flip(2,3)
        for i in range(self.depth):
            x = torch.nn.functional.conv2d(x, kernel[...,i], **self.kwargs)
        x = torch.fft.ifftshift(x, dim=(-2,-1))
        x = torch.fft.ifft2(x)
        x = x.real
        # x = torch.layer_norm(x, x.size()[1:])
        x = self.proj(x)
        x = res + 1e-3 * x
        return x


class DynamicGWConv2d(torch.nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        hidden_channels:int,
        kernel_size:int, 
        depth:int,
        classes:int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kernel_size%2==1, 'kernel_size must be odd'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.kwargs = kwargs
        self.kwargs['padding'] = self.kernel_size // 2

        self.emb = torch.nn.Conv2d(in_channels, hidden_channels, 1)
        self.proj = torch.nn.Conv2d(2*hidden_channels, out_channels, 1)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, depth*classes),
            torch.nn.Unflatten(-1, (depth, classes)),
            torch.nn.Softmax(dim=-1),
        )

        self.kernel = torch.nn.Parameter(
            torch.randn(2*hidden_channels, hidden_channels, kernel_size, kernel_size, 1, depth, classes, dtype=torch.cfloat)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform(self.kernel, a=math.sqrt(5), mode='fan_out')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        x = self.emb(x)
        x = torch.fft.fft2(x)
        attn = self.attn(x.real.select(-1, 0).select(-1, 0))
        x1 = x.mul(1j*torch.fft.fftfreq(x.size(-2), device=x.device).unsqueeze(-1))
        x2 = x.mul(1j*torch.fft.fftfreq(x.size(-1), device=x.device).unsqueeze(-2))
        x = torch.cat([x1, x2], dim=-3)
        x = torch.fft.fftshift(x, dim=(-2,-1))
        size = x.size()[:2]
        x = x.flatten(0, 1).unsqueeze(0)
        kernel = self.kernel + self.kernel.conj().flip(2,3)
        kernel = self.kernel.mul(attn.cfloat()).sum(-1)
        kernel = kernel.permute(4, 0, 1, 2, 3, 5).flatten(0, 1)
        for i in range(self.depth):
            x = torch.nn.functional.conv2d(x, kernel[...,i], groups=2*size[0],**self.kwargs)
        x = x.squeeze(0).unflatten(0, size)
        x = torch.fft.ifftshift(x, dim=(-2,-1))
        x = torch.fft.ifft2(x)
        x = x.real
        # x = torch.layer_norm(x, x.size()[1:])
        x = self.proj(x)
        x = res + 1e-3 * x
        return x


class GWConv1dBlock(torch.nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            hidden_channels:int,
            kernel_size:int, 
            depth:int, 
            classes:int = None,
            **kwargs,
        ) -> None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        if classes is None:
            self.conv = GWConv1d(in_channels, out_channels, hidden_channels, kernel_size, depth, **kwargs)
        else:
            self.conv = DynamicGWConv1d(in_channels, out_channels, hidden_channels, kernel_size, depth, classes, **kwargs)
        self.bias = torch.nn.Parameter(torch.randn(out_channels, 1))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.bias
        x = self.activation(x)
        return x
    

class GWConv2dBlock(torch.nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int, 
            hidden_channels:int,
            kernel_size:int, 
            depth:int,
            classes:int = None, 
            **kwargs,
        ) -> None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        if classes is None:
            self.conv = GWConv2d(in_channels, out_channels, hidden_channels, kernel_size, depth, **kwargs)
        else:
            self.conv = DynamicGWConv2d(in_channels, out_channels, hidden_channels, kernel_size, depth, classes, **kwargs)
        self.bias = torch.nn.Parameter(torch.randn(out_channels, 1, 1))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.bias
        x = self.activation(x)
        return x
    