from typing import Tuple, Union
import torch

from gw.base import cubic_interpolation, gw
from gw.utils import make_pad_mask


class GW(torch.nn.Module):
    def __init__(self, r: float = 1.0):
        """General warping module

        Args:
            r (float, optional): 
                Warping scale. Set large if the model only outputs alignments close to the diagonal. 
                If too far from the diagonal, make it smaller. Defaults to 1.0.
        """
        super(GW, self).__init__()
        self.r = r

    def _preprocess(self, w: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b = torch.arange(w.size(0), device=w.device)

        t = torch.arange(w.size(-1), device=w.device).unsqueeze(0).expand(w.size(0), -1)
        t = t / (lengths.unsqueeze(-1) - 1)
        t = t.unsqueeze(1)

        mask = make_pad_mask(lengths, w.size(-1))

        w = w.cumsum(-1)
        w = w - ((1-t)*w[:,:,0].unsqueeze(-1) + t*w[b,:,lengths-1].unsqueeze(-1))

        w = w / w.size(1) * self.r

        w = w.masked_fill(mask.unsqueeze(1), 0.0)

        return w

    def forward(
            self, 
            x: torch.Tensor, 
            w: torch.Tensor, 
            x_lengths: torch.Tensor,
            return_att_w = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward genaral warping

        Args:
            x (torch.Tensor): Input sequence to be warped (batch, channel, time).
            w (torch.Tensor): Warping parameters (batch, k, time). k in 4 ~ 16 is recommended.
            x_lengths (torch.Tensor): Length of input sequence (batch,).
            return_att_w (bool, optional): Return attention weight. Defaults to False.

        Returns:
            torch.Tensor: Warped sequence (batch, channel, time).
            torch.Tensor: Attention weight (batch, time_i, time_o). This is returned only if return_att_w is True.
        """
        w = self._preprocess(w, x_lengths)

        f = None
        for i in range(w.size(1)):
            f = gw(w[:,i], f)

        mask = make_pad_mask(x_lengths, x.size(-1))

        x = x.masked_fill(mask.unsqueeze(1), 0.0)

        y = cubic_interpolation(x, f)

        y = y.masked_fill(mask.unsqueeze(1), 0.0)

        if return_att_w:
            att_w = torch.eye(x.size(-1), device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
            att_w = cubic_interpolation(att_w, f)

            att_w = att_w.masked_fill(mask.unsqueeze(1), 0.0)
            att_w = att_w.masked_fill(mask.unsqueeze(-1), 0.0)

            return y, att_w

        return y
    