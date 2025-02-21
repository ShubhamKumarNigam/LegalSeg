# LegalSeg Rhetorical Role Classification - Data Repository

## Data Repository Link

[Data Repository on Onedrive](https://iitk-my.sharepoint.com/personal/sknigam_iitk_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsknigam%5Fiitk%5Fac%5Fin%2FDocuments%2FServer%5FData%2FLegalSeg%2FData&ga=1)

This repository contains the datasets used for training, validating, and testing various models for rhetorical role classification in legal documents. The datasets are provided in different formats (CSV, JSON, and folder-based) to suit the needs of each model. Below is a detailed explanation of the dataset organization and file structure.

## Folder Structure

```plaintext
data/
│
├── Hier_BiLSTM_CRF/
│   ├── test/                                # Test data folder containing 712 text files
│   └── train/                               # Training data folder containing 4984 text files
│   └── val/                                 # Validation data folder containing 1424 text files
|
├── InLegalBERT/
│   ├── test.csv                             # Test data for InLegalBERT/Role-Aware/GNN/Rhetoric-LLaMA
│   ├── train.csv                            # Training data for InLegalBERT/Role-Aware/GNN/Rhetoric-LLaMA
│   └── val.csv                              # Validation data for InLegalBERT/Role-Aware/GNN/Rhetoric-LLaMA
│
├── MTL/
│   ├── json/
│   │   ├── test.json                        # Test data for MTL in JSON format
│   │   ├── train.json                       # Training data for MTL in JSON format
│   │   └── val.json                         # Validation data for MTL in JSON format
│   └── mtl-data-002.zip                     # Zipped data for MTL
│
└── ToInLegalBERT/
    ├── test.json                            # Test data for ToInLegalBERT
    ├── train.json                           # Training data for ToInLegalBERT
    └── val.json                             # Validation data for ToInLegalBERT
