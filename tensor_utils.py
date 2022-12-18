import torch
from torch import Tensor


class TensorUtils:
    def get_mask(self, sz):
        """Generates a square subsequent mask as an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def batchify(self, data: Tensor, bsz: int, device='cpu') -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.
        Args:
            data: Tensor, shape [N]
            bsz: int, batch size
        Returns:
            Tensor of shape [N // bsz, bsz]
            """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)

    def get_batch(self, source, i, bptt):
        """source: Tensor, shape [full_seq_len, batch_size]
                i: int
        Returns:
            tuple (data, target)
            data.shape = [seq_len, batch_size]
            target.shape = [seq_len * batch_size]
        """
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len]
        #print(data)
        #print(target)
        return data, target.reshape(-1)