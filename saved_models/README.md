# Saved Models Folder Structure

This folder contains the saved models for each of the different architectures used in the rhetorical role classification task. The models are stored in various formats such as `.pt`, `.tar`, and `.safetensors`.

## Saved Models Repository Link
[Saved Models Repository Link](https://iitk-my.sharepoint.com/personal/sknigam_iitk_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsknigam%5Fiitk%5Fac%5Fin%2FDocuments%2FServer%5FData%2FLegalSeg%2Fsaved%20models&ga=1)

## Folder Structure

```plaintext
saved_models/
│
├── GNN/
│   └── GNN_best_model.pt                              # Best saved model for the GNN
│
├── Hier_BiLSTM_CRF/
│   ├── data_state4.json                               # Data state for the Hier_BiLSTM_CRF model
│   ├── model_state4.tar                               # Model state for the Hier_BiLSTM_CRF model
│   ├── tag2idx.json                                   # Tag-to-index mapping for the Hier_BiLSTM_CRF model
│   └── word2idx.json                                  # Word-to-index mapping for the Hier_BiLSTM_CRF model
│
├── InLegalBERT/
│   ├── InLegalBERT(i-1, i, i+1)model.safetensors      # Saved model for InLegalBERT (previous, current, next sentence)
│   ├── InLegalBERT(i-1, i)model.safetensors           # Saved model for InLegalBERT (previous and current sentence)
│   ├── InLegalBERT(i-1, label_t, i)model.safetensors  # Saved model for InLegalBERT (previous sentence with true label)
│   ├── InLegalBERT(i-2, i-1, i)model.safetensors      # Saved model for InLegalBERT (two previous sentences and current sentence)
│   ├── InLegalBERT(i)model.safetensors                # Saved model for InLegalBERT (current sentence)
│   └── InLegalBERT(pos, i)model.safetensors           # Saved model for InLegalBERT (positional)
│
├── MTL/
│   ├── model/
│   │   ├── data_state.json                            # Data state for MTL
│   │   ├── model_state.tar                            # Model state for MTL
│   │   ├── tag2idx.json                               # Tag-to-index mapping for MTL
│   │   ├── tag2idx_binary.json                        # Tag-to-index mapping (binary) for MTL
│   │   ├── word2idx.json                              # Word-to-index mapping for MTL
│   │   └── word2idx_binary.json                       # Word-to-index mapping (binary) for MTL
│   └── Shift_Embeddings_model/
│       ├── data_state.json                            # Data state for MTL Shift Embeddings
│       ├── model_state.tar                            # Model state for MTL Shift Embeddings
│       ├── tag2idx.json                               # Tag-to-index mapping for MTL Shift Embeddings
│       ├── tag2idx_binary.json                        # Tag-to-index mapping (binary) for MTL Shift Embeddings
│       ├── word2idx.json                              # Word-to-index mapping for MTL Shift Embeddings
│       └── word2idx_binary.json                       # Word-to-index mapping (binary) for MTL Shift Embeddings
│
├── RhetoricLLaMA/
│   ├── adapter_config.json                            # Adapter configuration for RhetoricLLaMA
│   ├── adapter_model.safetensors                      # Adapter model for RhetoricLLaMA
│   ├── README.md                                      # README for RhetoricLLaMA
│   ├── special_tokens_map.json                        # Special tokens mapping for RhetoricLLaMA
│   ├── tokenizer.json                                 # Tokenizer file for RhetoricLLaMA
│   └── tokenizer_config.json                          # Tokenizer configuration for RhetoricLLaMA
│
├── Role-Aware/
│   ├── Role-Aware/
│   │   ├── best_model.pt                              # Best model for Role-Aware
│   │   └── final_model.pth                            # Final model for Role-Aware
│   ├── Role-Aware_CI/
│   │   ├── best_model.pt                              # Best model for Role-Aware with Confidence Intervals
│   │   └── final_model.pth                            # Final model for Role-Aware with Confidence Intervals
│   └── Role-Aware_label_definitions/
│       ├── best_model.pt                              # Best model for Role-Aware with label definitions
│       └── final_model.pth                            # Final model for Role-Aware with label definitions
│
└── ToInLegalBERT/
    └── ToInlegalBERT.zip                      # ToInLegalBERT model
