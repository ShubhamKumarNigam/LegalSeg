import torch
import transformers


class TransformerOverBERT(torch.nn.Module):

    def __init__(
        self,
        bert,
        embedder,
        transformer,
        max_document_length,
        max_segment_length,
        device="cpu",
        **kwargs
    ):

        super().__init__(**kwargs)

        self._max_document_length = max_document_length
        self._max_segment_length = max_segment_length
        
        self._bert = bert
        self._embedder = embedder
        self._transformer = transformer
        
        self._device = device

    def forward(self, input_ids, **kwargs):

        n_documents = input_ids.size(0)

        # Padding segments mask
        # Shape: (n_documents, max_document_length)
        segments_mask = (input_ids != 0).any(-1).to(self._device)

        input_ids = input_ids.reshape(
            n_documents * self._max_document_length,
            self._max_segment_length
        )

        # BERT output
        # Shape: (n_documents * max_document_length, max_segment_length, bert_hidden_size)
        output = self._bert(input_ids)["last_hidden_state"]

        bert_hidden_size = output.size(2)
        output = output.reshape(
            n_documents,
            self._max_document_length,
            self._max_segment_length,
            bert_hidden_size
        )

        # Retrieve CLS
        # Shape: (n_documents, max_document_length, bert_hidden_size)
        output = output[:, :, 0]

        # Embedding
        # Shape: (n_documents, max_document_length, bert_hidden_size)
        output = self._embedder(x=output, mask=segments_mask)

        # Filter padding segments
        segments_mask = segments_mask.unsqueeze(2).expand(
            n_documents,
            self._max_document_length,
            bert_hidden_size
        )
        output *= segments_mask

        # Transformer output
        # Shape: (n_documents, max_document_length, bert_hidden_size)
        output = self._transformer(output)

        return transformers.file_utils.ModelOutput(
            last_hidden_state = output,
            past_key_values = None,
            hidden_states = output,
            attentions = None,
            cross_attentions = None
        )


def AutoTransformerOverBERTForTokenClassification(
    model_name,
    embedder,
    transformer,
    num_labels,
    max_document_length,
    max_segment_length,
    device="cpu"
):

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.bert = TransformerOverBERT(
        bert=model.bert,
        embedder=embedder,
        transformer=transformer,
        max_document_length=max_document_length,
        max_segment_length=max_segment_length,
        device=device
    )
    return model
