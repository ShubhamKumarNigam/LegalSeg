from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class Embedder(ABC):
    def __init__(self, embedding_dimension):
        self._embedding_dimension = embedding_dimension

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    @abstractmethod
    def forward(self, x, **kwargs):
        pass


class IdemEmbedder:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        return x


class SinusoidalEmbedder(Embedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_embeddings(self, **kwargs):
        sinusoidal_embedding = torch.Tensor(
            [
                [
                    w / pow(10_000, 2 * i / self._embedding_dimension)
                    for i in range(self._embedding_dimension)
                ]
                for w in self._get_positions_weights(**kwargs)
            ]
        )

        sinusoidal_embedding[:, 0::2] = torch.sin(sinusoidal_embedding[:, 0::2])
        sinusoidal_embedding[:, 1::2] = torch.cos(sinusoidal_embedding[:, 1::2])

        return sinusoidal_embedding

    @abstractmethod
    def _get_positions_weights(self, **kwargs):
        pass


class AbsoluteSinusoidalEmbedder(SinusoidalEmbedder):
    def __init__(self, max_document_length, **kwargs):
        super().__init__(**kwargs)
        self._max_document_length = max_document_length
        self._embeddings = self._get_embeddings()

    def _get_positions_weights(self, **kwargs):
        return torch.arange(1, self._max_document_length + 1)

    def forward(self, x, **kwargs):
        return x + self._embeddings.to(x.device)


class RelativeSinusoidalEmbedder(SinusoidalEmbedder):
    def __init__(self, max_document_length, **kwargs):
        super().__init__(**kwargs)
        self._max_document_length = max_document_length
        self._embeddings_lookup_table = self._get_embeddings_lookup_table()

    def _get_embeddings_lookup_table(self):
        embeddings_lookup_table = {}
        for document_length in tqdm(range(1, self._max_document_length + 1)):
            embeddings_lookup_table[document_length] = self._get_embeddings(
                document_length=document_length
            )
        return embeddings_lookup_table

    def _get_positions_weights(self, document_length, **kwargs):
        return 1_000 * torch.arange(1, self._max_document_length + 1) / document_length

    def forward(self, x, mask, **kwargs):
        documents_length = torch.count_nonzero(mask, 1)
        embeddings = torch.stack(
            [
                self._embeddings_lookup_table[document_length.item()]
                for document_length in documents_length
            ]
        )
        return x + embeddings.to(x.device)
