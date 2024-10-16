import json

import pandas as pd
import torch
import transformers


class RhetoricalRolesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_filepath,
        tokenizer_model_name,
        max_segment_length,
        has_labels=True,
    ):
        self._df = pd.read_csv(data_filepath)

        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)
        self._inputs = torch.as_tensor(
            tokenizer(
                self._df.segments.tolist(),
                padding="max_length",
                max_length=max_segment_length,
                truncation=True,
            )["input_ids"]
        )

        self._has_labels = has_labels
        if self._has_labels:
            self._labels = torch.as_tensor(
                self._df.labels.tolist()
            ).long()  # Loss computation requires labels of type long

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return (self._inputs[idx], self._labels[idx]) if self._has_labels else self._inputs[idx]


class RhetoricalRolesDatasetForTransformerOverBERT(torch.utils.data.Dataset):
    def __init__(
        self,
        data_filepath,
        max_document_length,
        max_segment_length,
        tokenizer_model_name,
        has_labels=True,
    ):
        with open(data_filepath, "r") as df:
            self._documents = json.load(df)
        self._has_labels = has_labels

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)

        self._max_document_length = max_document_length
        self._max_segment_length = max_segment_length

        self._inputs = self._get_inputs()
        if self._has_labels:
            self._labels = self._get_labels()

    def _get_inputs(self):
        documents_input_ids = []
        for document in self._documents:
            document_input_ids = self._tokenizer(
                document["segments"],
                padding="max_length",
                truncation=True,
                max_length=self._max_segment_length,
            )["input_ids"]
            document_input_ids += [
                [0] * self._max_segment_length
            ] * self._get_document_padding_length(document)
            document_input_ids = torch.as_tensor(document_input_ids)
            documents_input_ids.append(document_input_ids)

        return documents_input_ids

    def _get_labels(self):
        documents_labels = []
        for document in self._documents:
            document_labels = document["labels"] + [-100] * self._get_document_padding_length(
                document
            )
            document_labels = torch.as_tensor(
                document_labels
            ).long()  # Loss computation requires labels of type long
            documents_labels.append(document_labels)

        return documents_labels

    def _get_document_padding_length(self, document):
        return self._max_document_length - len(document["segments"])

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return (self._inputs[idx], self._labels[idx]) if self._has_labels else self._inputs[idx]
