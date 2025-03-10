# Output Folder Structure

This folder contains the predicted CSV files for each model used in the rhetorical role classification task. The outputs are stored as CSV files, with predictions for test datasets generated by each respective model.

## Huggingface Output/Predictions Link 

[Output/Predictions Link](https://huggingface.co/collections/L-NLProc/legalseg-predictions-67b835db622b213fdfebc357)

## Folder Structure

```plaintext
output/
│
├── GNN/
│   └── gnn_predictions.csv                 # Predictions from the GNN model
│
├── Hier_BiLSTM_CRF/
│   ├── hier_bilstm_crf_predictions.csv      # Predictions from the Hierarchical BiLSTM-CRF model
│   └── predictions.txt                      # Only contains the predicted labels
│
├── InLegalBERT/
│   ├── inlegalbert_i_predictions.csv        # Predictions from InLegalBERT (current sentence only)
│   ├── inlegalbert_i-1_i_predictions.csv    # Predictions from InLegalBERT (previous and current sentence)
│   ├── inlegalbert_i-1_i_i+1_predictions.csv# Predictions from InLegalBERT (previous, current, and next sentence)
│   ├── inlegalbert_i-1_label_p_predictions.csv # Predictions from InLegalBERT (previous sentence with predicted label)
│   ├── inlegalbert_i-1_label_t_predictions.csv # Predictions from InLegalBERT (previous sentence with true label)
│   └── inlegalbert_pos_i_predictions.csv    # Predictions from InLegalBERT (positional model)
│
├── MTL/
│   └── mtl_predictions.csv                  # Predictions from the MTL model
│
├── RhetoricLLaMA/
│   └── RhetoricLLaMA_predictions.csv        # Predictions from the RhetoricLLaMA model
│
├── Role-Aware/
│   ├── role_aware_predictions.csv           # Predictions from the Role-Aware Transformers model
|   ├── predictions_CI.csv                   # Predictions from the Role-Aware Class Inbalence Transformers model
|   └── role-aware_label_definations_predictions.csv # Predictions from the Role-Aware label defination predictions Transformers model
│
└── ToInLegalBERT/
    └── toinlegalbertpredictions.csv         # Predictions from the ToInLegalBERT model
```
