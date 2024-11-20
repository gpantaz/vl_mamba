import random

from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer


class FakeDataset(Dataset[BatchEncoding]):
    """This is a simple dataset to benchmark the Longformer architecture.

    It samples a dataset of an arbitrary size that generates sequences of tokens of a fixed length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_size: int,
        text_seqlen: int = 20,
        video_seqlen: int = 60,
        num_objects: int = 18,
    ):
        self.video_seqlen = video_seqlen
        self.text_seqlen = text_seqlen
        self.dataset_size = dataset_size
        self.num_objects = num_objects
        self.tokenizer = tokenizer
        self.fake_string = """
        this is a very long caption that describes
        the current video where multiple persons are playing football."
        """

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> BatchEncoding:
        """Generate fake video features."""
        tokens = [self.tokenizer.cls_token]

        sequence_length = random.randint(5, self.video_seqlen)

        for _ in range(sequence_length):
            next_sequence = [self.tokenizer.sep_token for _ in range(self.num_objects)]
            tokens.extend(next_sequence)
            tokens.append(self.tokenizer.sep_token)

        tokens.extend(self.tokenizer.tokenize(self.fake_string))

        text = " ".join(tokens)

        # 'text' is a fake string that contains both "visual tokens" and "text tokens"
        return self.tokenizer(text)
