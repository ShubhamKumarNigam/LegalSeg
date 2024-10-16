# LegalSeg Rhetorical Role Classification - Data Repository
# [Dataset Repository Link](https://drive.google.com/file/d/1my359TGjmDDzAf9lh0zcL-PgCcnQd_Z3/view?usp=sharing)
This repository contains the datasets used for training, validating, and testing various models for rhetorical role classification in legal documents. The datasets are provided in different formats (CSV, JSON, and folder-based) to suit the needs of each model. Below is a detailed explanation of the dataset organization and file structure.

## Repository Structure

### CSV Files

These CSV files are used by the following models: **GNN**, **Role-Aware**, **all InLegalBERT variants**, and **RhetoricLLaMA**.



### JSON Files

These JSON files are used by the following models: **MTL** and **ToInLegalBERT**.


### Folder-based Structure

The **Hier_BiLSTM_CRF** model uses a folder-based structure with separate files for each document in both the training and test sets.


## Dataset Description

The datasets in this repository consist of legal judgments, each annotated with rhetorical role labels. They are organized in various formats (CSV, JSON, folder-based) to match the needs of different models in this repository:

- **CSV Files**: Used by **GNN**, **Role-Aware Transformers**, **InLegalBERT variants**, and **RhetoricLLaMA**. These CSV files contain structured data where each row represents a sentence or document, along with its corresponding rhetorical role label.
  
- **JSON Files**: Used by **MTL** and **ToInLegalBERT** models. The JSON format is more flexible, allowing hierarchical data structures that are useful for models requiring complex input/output relationships.

- **Folder-based Structure**: The **Hier_BiLSTM_CRF** model uses separate files for each document. These documents are stored in distinct folders for training and testing.

The **RhetoricLLaMA** model also includes an additional CSV file, `instructions_decision.csv`, which contains the instruction sets used for instruction-tuning.
