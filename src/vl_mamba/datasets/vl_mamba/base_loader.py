from abc import abstractmethod
from collections.abc import Iterator
from functools import partial
from multiprocessing import Pool
from typing import Any, Optional

import datasets
import pyarrow as pa

from vl_mamba.datamodels.datamodels import DatasetFeatures


class BaseLoader:
    """Base class for all loaders."""

    def __init__(
        self,
        source: str,
        split: str,
        writer_batch_size: int,
        chunk_size: int = 1,
        num_proc: int = 1,
    ) -> None:
        self.source = source
        self.split = split
        self.writer_batch_size = writer_batch_size
        self.chunk_size = chunk_size
        self.num_proc = num_proc
        self.gen_kwargs: dict[str, Any] = {}

    @abstractmethod
    def _generate_examples(
        self, examples: list[Any], **kwargs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        raise NotImplementedError()

    @abstractmethod
    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        raise NotImplementedError()

    def _generate_batches(self) -> Iterator[pa.Table]:
        rows_iterator = self._build_rows_iterator(chunk_size=self.chunk_size, **self.gen_kwargs)
        if self.num_proc == 1:
            for row in rows_iterator:  # noqa: WPS526
                yield self._generate_tables(row, **self.gen_kwargs)
        else:
            with Pool(self.num_proc) as pool:
                tables_iterator = pool.imap(
                    partial(self._generate_tables, **self.gen_kwargs),
                    rows_iterator,
                    chunksize=1,
                )
                yield from tables_iterator


class DatasetsLoader(BaseLoader):
    """Helper as some datasets are already implemented."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        num_proc: int,
        config_name: Optional[str] = None,
        datasets_batch_size: int = 1000,
        streaming: bool = False,
    ):
        super().__init__(source=dataset_name, split=split, writer_batch_size=datasets_batch_size)
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.num_proc = num_proc
        self.datasets_batch_size = datasets_batch_size
        self.streaming = streaming

    @abstractmethod
    def cast_to_vlmamba_features(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Return list of rows casted as MMICL features."""
        raise NotImplementedError()

    def _generate_examples(
        self, examples: list[Any], **kwargs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        batch: dict[str, list[Any]] = {}
        for key in examples[0]:
            batch[key] = []

        for example in examples:
            for example_key, example_value in example.items():
                batch[example_key].append(example_value)

        return batch

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self.cast_to_vlmamba_features(self._generate_examples(examples, **kwargs))
        return pa.table(DatasetFeatures.encode_batch(output_batch))


class BaseLoaderWithDLManager(BaseLoader):
    """We use dl_manager to generate `gen_kwargs` needed in order to generate examples."""

    def __init__(
        self,
        dl_manager: datasets.DownloadManager,
        source: str,
        split: str,
        num_proc: int,
        chunk_size: int,
        writer_batch_size: int = 10000,
    ):
        super().__init__(source=source, split=split, writer_batch_size=writer_batch_size)
        self.gen_kwargs = self.generate_gen_kwargs(dl_manager)
        # Used for multiprocessing
        self.chunk_size = chunk_size
        self.num_proc = num_proc

    @abstractmethod
    def generate_gen_kwargs(self, dl_manager: datasets.DownloadManager) -> dict[str, Any]:
        """Generate keyword arguments needed to generate examples."""
        raise NotImplementedError()

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        return pa.table(DatasetFeatures.encode_batch(self._generate_examples(examples, **kwargs)))
