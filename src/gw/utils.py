import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Make mask tensor for padding

    Args:
        lengths (torch.Tensor): Lengths of each sequence (batch,).
        max_len (int, optional): Maximum length of sequences. Defaults to None.

    Returns:
        torch.Tensor: Mask tensor for padding (batch, max_len).
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(-1))
    return mask

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Make mask tensor for non-padding

    Args:
        lengths (torch.Tensor): Lengths of each sequence (batch,).
        max_len (int, optional): Maximum length of sequences. Defaults to None.

    Returns:
        torch.Tensor: Mask tensor for non-padding (batch, max_len).
    """
    return ~make_pad_mask(lengths, max_len)