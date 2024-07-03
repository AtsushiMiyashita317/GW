# General Warping group implementation

## Install

```bash
pip install git+https://github.com/AtsushiMiyashita317/GW.git
```

## Usage

```python
from gw import GW


xs: torch.Tensor        # Input to be warped (batch, channel, time).
x_lengths: torch.Tensor # Length of input sequence (batch,).
ws: torch.Tensor        # Warping parameters (batch, k, time). k in 4~16 is recommended.

warp = GW()
# You can use any differentiable module as warping predictor
warping_predictor = torch.nn.Sequential(
    torch.nn.Conv1d(...),
    torch.nn.ReLU(),
    torch.nn.Conv1d(...),
    ...
)

ws = warping_predictor(xs)
# wWrping. Output size is (batch, channel, time)
ys = warp(xs, ws, x_lengths)
# When you want arignment map (batch, time_in, time_out)
ys, att_ws = warp(xs, ws, x_lengths, return_att_w = True)

```
