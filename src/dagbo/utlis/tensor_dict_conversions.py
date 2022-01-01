import torch
from torch import Tensor


def unpack_to_dict(names: list[str], values: Tensor) -> dict[str, Tensor]:
    """
    Args:
        names: d-length list
        values: batch_shape * d-dim Tensor
    Returns: d-length dict[name, value: batch_shape-dim Tensor]
    """
    split_values = torch.unbind(input=values, dim=-1)
    return {name: value for (name, value) in zip(names, split_values)}


def pack_to_tensor(names: list[str], values: dict[str, Tensor]) -> Tensor:
    """
    Args:
        names: d-length list
        values: d-length dict[name, value: batch_shape-dim Tensor]
    Returns: batch_shape * q * len(names)
        d is has the same order as `names`
    """
    return torch.stack(
        [values[y] for y in names],
        dim=-1)  # stack same shape tensor, with an added dimension
