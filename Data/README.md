# LegalSeg Rhetorical Role Classification - Data Repository

## Data Repository Link

[Data Repository on Google Drive](https://drive.google.com/file/d/1my359TGjmDDzAf9lh0zcL-PgCcnQd_Z3/view?usp=sharing)

This repository contains the datasets used for training, validating, and testing various models for rhetorical role classification in legal documents. The datasets are provided in different formats (CSV, JSON, and folder-based) to suit the needs of each model. Below is a detailed explanation of the dataset organization and file structure.

## Folder Structure

```plaintext
data/
│
├── GNN/
│   ├── test.csv                             # Test data for GNN model
│   ├── train.csv                            # Training data for GNN model
│   └── val.csv                              # Validation data for GNN model
│
├── Hier_BiLSTM_CRF/
│   ├── test/                                # Test data folder containing 712 text files
│   └── train/                               # Training data folder containing 4984 text files
│
├── InLegalBERT/
│   ├── test.csv                             # Test data for InLegalBERT
│   ├── train.csv                            # Training data for InLegalBERT
│   └── val.csv                              # Validation data for InLegalBERT
│
├── MTL/
│   ├── json/
│   │   ├── test.json                        # Test data for MTL in JSON format
│   │   ├── train.json                       # Training data for MTL in JSON format
│   │   └── val.json                         # Validation data for MTL in JSON format
│   └── mtl-data-002.zip                     # Zipped data for MTL
│
├── RhetoricLLaMA/
│   ├── instructions_decision.csv            # Instruction sets for RhetoricLLaMA model fine-tuning
│   ├── IT_CL.csv                            # Additional data for RhetoricLLaMA model
│   ├── test.csv                             # Test data for RhetoricLLaMA
│   ├── test_Build.csv                       # Test build data for RhetoricLLaMA
│   ├── train.csv                            # Training data for RhetoricLLaMA
│   └── val.csv                              # Validation data for RhetoricLLaMA
│
├── Role-Aware/
│   ├── test.csv                             # Test data for Role-Aware Transformers
│   ├── train.csv                            # Training data for Role-Aware Transformers
│   └── val.csv                              # Validation data for Role-Aware Transformers
│
└── ToInLegalBERT/
    ├── test.json                            # Test data for ToInLegalBERT
    ├── train.json                           # Training data for ToInLegalBERT
    └── val.json                             # Validation data for ToInLegalBERT
