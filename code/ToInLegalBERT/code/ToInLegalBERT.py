import joblib
import os
import optuna
import sys
import torch
import wandb

sys.path.insert(0, os.path.abspath('../'))

from rhetorical_roles_classification import (
    AbsoluteSinusoidalEmbedder,
    AutoTransformerOverBERTForTokenClassification,
    RelativeSinusoidalEmbedder,
    RhetoricalRolesDatasetForTransformerOverBERT
)
from rhetorical_roles_classification.train import train
from rhetorical_roles_classification.test import test_ToBERT

from config import (
    DATA_FOLDER,
    MODELS_FOLDER,
    HUGGING_FACE_BERT_MODEL,
    NUM_LABELS,
    WANDB_PROJECT,
    MAX_SEGMENT_LENGTH,
    label2rhetRole
)

MODEL_NAME = "ToInlegalBERT"

MAX_DOCUMENT_LENGTH = 601

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

wandb_login_success = wandb.login(key = 'ef38ca12623812f96a8401b6724c64aeccd8f3f3')

train_dataset = RhetoricalRolesDatasetForTransformerOverBERT(
    data_filepath=os.path.join(DATA_FOLDER, "train.json"),
    max_document_length=MAX_DOCUMENT_LENGTH,
    max_segment_length=MAX_SEGMENT_LENGTH,
    tokenizer_model_name=HUGGING_FACE_BERT_MODEL
)
valid_dataset = RhetoricalRolesDatasetForTransformerOverBERT(
    data_filepath=os.path.join(DATA_FOLDER, "valid.json"),
    max_document_length=MAX_DOCUMENT_LENGTH,
    max_segment_length=MAX_SEGMENT_LENGTH,
    tokenizer_model_name=HUGGING_FACE_BERT_MODEL
)
test_dataset = RhetoricalRolesDatasetForTransformerOverBERT(
    data_filepath=os.path.join(DATA_FOLDER, "test.json"),
    max_document_length=MAX_DOCUMENT_LENGTH,
    max_segment_length=MAX_SEGMENT_LENGTH,
    tokenizer_model_name=HUGGING_FACE_BERT_MODEL
)

absolute_embedder = AbsoluteSinusoidalEmbedder(
    max_document_length=MAX_DOCUMENT_LENGTH,
    embedding_dimension=768
)
joblib.dump(absolute_embedder, os.path.join(MODELS_FOLDER, "absolute_embedder"))

relative_embedder = RelativeSinusoidalEmbedder(
    max_document_length=MAX_DOCUMENT_LENGTH,
    embedding_dimension=768
)
joblib.dump(relative_embedder, os.path.join(MODELS_FOLDER, "relative_embedder"))

absolute_embedder = joblib.load(os.path.join(MODELS_FOLDER, "absolute_embedder"))
relative_embedder = joblib.load(os.path.join(MODELS_FOLDER, "relative_embedder"))

hyperparameters = {
    "batch_size": 1,
    "epochs": 3,
    "num_warmup_steps": 0,
    "accum_iter": 3,
    "early_stopping": True,
    "early_stopping_patience": 2
}

def objective(trial):
    
    print("\n************************ Starting New Optuna Trial ************************\n")
    
    hyperparameters["lr"] = trial.suggest_float("lr", 5e-6, 5e-4)
    hyperparameters["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 1e-1)
    
    embedding_type = trial.suggest_categorical("embedding", ["abs", "rel"])
    embedder = absolute_embedder if embedding_type == "abs" else relative_embedder

    transformer_dropout = trial.suggest_float("transformer_dropout", 0.1, 0.7)
    transformer_dim_feedforward = trial.suggest_int("transformer_dim_feedforward", 50, 1000)
    transformer = torch.nn.Transformer(
        d_model=768,
        nhead=12,
        batch_first=True,
        dim_feedforward=transformer_dim_feedforward,
        activation="gelu",
        dropout=transformer_dropout,
        layer_norm_eps=1e-12,
        num_encoder_layers=2,
        num_decoder_layers=0
    ).encoder

    model = AutoTransformerOverBERTForTokenClassification(
        model_name=HUGGING_FACE_BERT_MODEL,
        embedder=embedder,
        transformer=transformer,
        num_labels=NUM_LABELS,
        max_document_length=MAX_DOCUMENT_LENGTH,
        max_segment_length=MAX_SEGMENT_LENGTH,
        device=DEVICE
    )
    model_id = f"{MODEL_NAME}-Trial{trial.number}"

    with wandb.init(project=WANDB_PROJECT, config=hyperparameters, name=model_id):

        wandb.config["embedding_type"] = embedding_type
        wandb.config["transformer_dropout"] = transformer_dropout
        wandb.config["transformer_dim_feedforward"] = transformer_dim_feedforward        

        valid_loss = train(
            model=model,
            destination_path=os.path.join(MODELS_FOLDER, model_id),
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=DEVICE,
            optuna_trial=trial,
            use_wandb=True,
            **hyperparameters
        )
        
    return valid_loss

study_name = f"{MODEL_NAME}.optuna"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize")

#study.optimize(objective, n_trials=32)

best_trial = study.best_trial
best_params = best_trial.params

embedder = absolute_embedder if best_params["embedding"] == "abs" else relative_embedder

transformer = torch.nn.Transformer(
    d_model=768,
    nhead=12,
    batch_first=True,
    dim_feedforward=best_params["transformer_dim_feedforward"],
    activation="gelu",
    dropout=best_params["transformer_dropout"],
    layer_norm_eps=1e-12,
    num_encoder_layers=2,
    num_decoder_layers=0
).encoder

model = AutoTransformerOverBERTForTokenClassification(
    model_name=HUGGING_FACE_BERT_MODEL,
    embedder=embedder,
    transformer=transformer,
    num_labels=NUM_LABELS,
    max_document_length=MAX_DOCUMENT_LENGTH,
    max_segment_length=MAX_SEGMENT_LENGTH,
    device=DEVICE
)

model_id = f"{MODEL_NAME}-Trial{best_trial.number}"
model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, model_id)))

test_ToBERT(
    model=model,
    test_dataset=test_dataset,
    label2rhetRole=label2rhetRole,
    device=DEVICE,
)
