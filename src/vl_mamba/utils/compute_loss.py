import torch
from torch.nn import CrossEntropyLoss


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """Returns a moderately tiny value for a given PyTorch data type.

    This is used to avoid numerical issues such as division by zero. This is different from
    `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs. Only supports floating point
    dtypes. Implementation from AllenNLP:
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L2010-L2024
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in {torch.float, torch.double}:
        return 1e-13  # noqa: WPS432
    elif dtype == torch.half:
        return 1e-4  # noqa: WPS432
    raise TypeError(f"Does not support dtype {str(dtype)}")


def masked_mean(
    vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """To calculate mean along certain dimensions on masked values.

    Implementation from AllenNLP:
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L351-L377
    Args:
        vector (torch.Tensor): The vector to calculate mean.
        mask (torch.Tensor): The mask of the vector. It must be broadcastable with vector.
        dim (int): The dimension to calculate mean
        keepdim (bool): Whether to keep dimension
    Returns:
        (torch.Tensor): Masked mean tensor
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)  # noqa: WPS358

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def average_task_loss(labels: torch.Tensor, logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Average the loss with respect to the sequence length for each element in the batch.

    This is used so that loss for smaller sequences is not penalized:
    1) Compute the cross-entropy loss a
    2) Average by sequence length
    3) Average by batch size
    """
    (bsz, seq_len) = labels.size()
    loss_fct = CrossEntropyLoss(reduction="none")

    labels_mask = labels != -100
    # flat_labels shape (batch_size, seq_len) -> (batch_size * seq_len)
    flat_labels = labels.view(-1)
    # flat_logits shape (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    flat_logits = logits.view(-1, vocab_size)
    # loss shape (batch_size, seq_len)
    loss = loss_fct(flat_logits, flat_labels).view(bsz, seq_len)
    # averages over the sequence length dimension first and then over the batch dimension
    return masked_mean(loss, labels_mask, dim=-1).mean()


def compute_loss(labels: torch.tensor, logits: torch.tensor, vocab_size: int) -> torch.tensor:
    """Compute loss averaged across the sequence length."""
    labels = labels.to(logits.device)  # type: ignore[assignment]
    logits = logits[:, -labels.size(1) :, :]
    # Shift so that tokens < n predict n and enable model parallelism
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)

    # Compute the loss averaged over the sequence length for each element in the batch
    loss = average_task_loss(shift_labels, shift_logits, vocab_size)
    return loss
