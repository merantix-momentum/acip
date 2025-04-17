from abc import ABC, abstractmethod
from typing import TypeAlias

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from acip.model.tokenizer_factory import TokenizerFactory

DatasetTypes: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


class DatasetFactory(ABC):
    """
    Base class for factories of Huggingface datasets.
    The concept of lazy creation is useful in PyTorch Lightning's DataModule, where the dataset is created only
    when needed. It also provides an interface `process_for_model_input` for preprocessing samples so that
    they can be directly fed to a `PreTrainedModel` by a PyTorch `DataLoader`.
    """

    def __init__(self, tokenizer_factory: TokenizerFactory):
        """
        Args:
            tokenizer_factory: A tokenizer (factory) is typically required to process samples of the dataset in
                `self.process_for_model_input`.
        """
        self.tokenizer_factory = tokenizer_factory

    @abstractmethod
    def create_dataset(self) -> DatasetTypes:
        """Creates the dataset."""
        raise NotImplementedError

    def __call__(self) -> DatasetTypes:
        """Creates the dataset."""
        return self.create_dataset()

    def process_for_model_input(self, dataset: DatasetTypes) -> DatasetTypes:
        """Preprocesses the dataset so that it can be directly fed to a `PreTrainedModel`."""
        return dataset


# Default tokenizer encoding keyword arguments used by `DefaultDatasetFactory`.
DEFAULT_TOKENIZER_ENCODING_KWARGS = {
    "max_length": 1024,
    "truncation": True,
    "padding": "max_length",
    "return_tensors": "pt",
}


class DefaultDatasetFactory(DatasetFactory):
    def __init__(
        self,
        tokenizer_factory: TokenizerFactory,
        seed: int | None = None,
        shuffle: bool = True,
        split: str | None = None,
        cache_dir: str | None = None,
        tokenizer_encoding_kwargs: dict | None = None,
    ):
        """
        Args:
            tokenizer_factory: Required for preprocessing samples in `self.process_for_model_input`.
            seed: Shuffle seed to be used.
            shuffle: Whether to shuffle the dataset or not.
            split: Which split of the Huggingface dataset to use. Value must be one of `self.available_splits`.
            cache_dir: Huggingface cache directory. Leave as `None` to use the default cache directory.
            tokenizer_encoding_kwargs: Keyword arguments to be passed to the tokenizer when used in
                `self.process_for_model_input`. Defaults to `DEFAULT_TOKENIZER_ENCODING_KWARGS`.
        """
        super().__init__(tokenizer_factory)

        self.tokenizer = self.tokenizer_factory()

        if tokenizer_encoding_kwargs is None:
            self.tokenizer_encoding_kwargs = DEFAULT_TOKENIZER_ENCODING_KWARGS
        else:
            self.tokenizer_encoding_kwargs = tokenizer_encoding_kwargs

        if split is None:
            self.split = self.available_splits[0]
        else:
            if split not in self.available_splits:
                raise ValueError(f"Split must be one of {self.available_splits}")
            self.split = split
        self.seed = seed
        self.shuffle = shuffle
        self.cache_dir = cache_dir

    def _tokenize(self, example):
        """Helper function for `self.process_for_model_input` that tokenizes a sample."""
        return self.tokenizer(example["text"], **self.tokenizer_encoding_kwargs)

    def _create_labels(self, batch):
        """Helper function for `self.process_for_model_input` that creates labels for a batch of samples."""
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    @property
    @abstractmethod
    def available_splits(self) -> list[str]:
        """Returns the available splits of the Huggingface dataset."""
        raise NotImplementedError


class C4DatasetFactory(DefaultDatasetFactory):
    """
    Factory for the C4 dataset (allennai/c4).
    """

    @property
    def available_splits(self) -> list[str]:
        return ["train", "validation"]

    def create_dataset(self) -> IterableDataset:
        return load_dataset("allenai/c4", "en", split=self.split, streaming=True, cache_dir=self.cache_dir)

    def process_for_model_input(self, dataset: IterableDataset) -> IterableDataset:
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        return dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        ).map(self._create_labels, batched=False)


class Wikitext2DatasetFactory(DefaultDatasetFactory):
    """
    Factory for the Wikitext-2 dataset (wikitext/wikitext-2-raw-v1).
    """

    @property
    def available_splits(self) -> list[str]:
        return ["train", "validation", "test"]

    def create_dataset(self) -> IterableDataset:
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split, streaming=True, cache_dir=self.cache_dir)

    def process_for_model_input(self, dataset: IterableDataset) -> IterableDataset:
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        return dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=["text"],
        ).map(self._create_labels, batched=False)


class GSM8KDatasetFactory(DefaultDatasetFactory):
    """
    Factory for the GSM8K dataset (openai/gsm8k).
    """

    @property
    def available_splits(self) -> list[str]:
        return ["train", "test"]

    def create_dataset(self) -> IterableDataset:
        return load_dataset("openai/gsm8k", "main", split=self.split, streaming=True, cache_dir=self.cache_dir)

    def process_for_model_input(self, dataset: IterableDataset) -> IterableDataset:
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        def _preprocess_data(example):
            """Helper function for `self.process_for_model_input` that converts Q&A pairs into a single string."""
            return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

        return (
            dataset.map(_preprocess_data)
            .map(self._tokenize, batched=True, remove_columns=["question", "answer", "text"])
            .map(self._create_labels, batched=False)
        )


class CoNaLaDatasetFactory(DefaultDatasetFactory):
    """
    Factory for the CoNaLa dataset (neulab/conala).
    """

    @property
    def available_splits(self) -> list[str]:
        return ["train"]

    def create_dataset(self) -> IterableDataset:
        return load_dataset("neulab/conala", "mined", split=self.split, streaming=True, cache_dir=self.cache_dir)

    def process_for_model_input(self, dataset: IterableDataset) -> IterableDataset:
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        def _preprocess_data(example):
            """Helper function for `self.process_for_model_input` that converts Q&A pairs into a single string."""
            return {"text": f"Intent: {example['intent']}\nSnippet: {example['snippet']}"}

        return (
            dataset.map(_preprocess_data)
            .map(
                self._tokenize,
                batched=True,
                remove_columns=["question_id", "intent", "snippet", "parent_answer_post_id", "id", "prob"],
            )
            .map(self._create_labels, batched=False)
        )
