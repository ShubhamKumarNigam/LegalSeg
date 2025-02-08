# LegalSeg Rhetorical Role Classification

This repository contains the implementation of several models for rhetorical role classification in legal documents for training and evaluation. Each model is implemented in its own folder with the corresponding training, inference, and data preparation scripts.


# Code Repository Structure

The following is the structure of the code repository for rhetorical role classification using various models:

```plaintext
LegalSeg_Rhetorical_Role_Classification/
│
├── GNN/
│   └── GNN.py                                 # Graph Neural Network implementation for rhetorical role classification
│
├── Hier_BiLSTM_CRF/
│   └── model/
│       ├── Hier_BiLSTM_CRF.py                 # Hierarchical BiLSTM-CRF model definition
│       ├── submodels.py                       # Sub-models for Hierarchical BiLSTM-CRF
│   └── infer.py                               # Inference script for Hier_BiLSTM_CRF
│   └── prepare_data.py                        # Data preparation for Hier_BiLSTM_CRF
│   └── run.py                                 # Script to run the training and evaluation
│   └── train.py                               # Training script for Hier_BiLSTM_CRF
│
├── InLegalBERT/
│   ├── InLegalBERT(i).py                      # InLegalBERT model with the current sentence as input
│   ├── InLegalBERT(i-1, i).py                 # InLegalBERT model with the previous and current sentence as input
│   ├── InLegalBERT(i-1, i, i+1).py            # InLegalBERT model with the previous, current, and next sentence as input
│   ├── InLegalBERT(i-1, label_p, i).py        # InLegalBERT model with the previous sentence and predicted label
│   ├── InLegalBERT(i-1, label_t, i).py        # InLegalBERT model with the previous sentence and true label
│   ├── InLegalBERT(i-2, i-1, i).py            # InLegalBERT model with two previous sentences and the current sentence
│   └── InLegalBERT(pos, i).py                 # InLegalBERT model with positional information
│
├── MTL/
│   ├── MTL.py                                 # Multi-task Learning model for rhetorical role classification
│   ├── embeddings_generation.py               # Embedding generation for MTL
│   └── Shift_embeddings.py                    # Shift embeddings script for MTL
│
├── RhetoricLLaMA/
│   ├── inference.py                           # Inference script for RhetoricLLaMA
│   └── instruction_fine-tune.py               # Instruction fine-tuning script for RhetoricLLaMA
│
├── Role-Aware/
│   ├── Role-Aware.py                          # Role-Aware Transformers implementation
│   └── Role-Aware_label_definitions.py        # Role-Aware Transformers with label definitions
│
└── ToInLegalBERT/
    └── code/
        ├── config.py                          # Configuration file for ToInLegalBERT
        └── ToInLegalBERT.py                   # TransformerOverInLegalBERT model implementation
    └── rhetorical_roles_classification/
        ├── embedding.py                       # Embedding generation for ToInLegalBERT
        ├── metrics_tracker.py                 # Metrics tracking during training and evaluation
        ├── rhetorical_roles_dataset.py        # Dataset processing for ToInLegalBERT
        ├── test.py                            # Test script for ToInLegalBERT
        ├── train.py                           # Training script for ToInLegalBERT
        └── transformer_over_bert.py           # TransformerOverInLegalBERT model script
