DATASET_NAME = "BUILD"

DATA_FOLDER = "../../data/ToInLegalBERT/"
MODELS_FOLDER = "../../saved_models/ToInLegalBERT/"  # We provide the best models only

HUGGING_FACE_BERT_MODEL = "law-ai/InLegalBERT"

WANDB_PROJECT = f"RhetoricalRolesClassification-{DATASET_NAME}"

rhetRole2label = {
    "None": 0,
    "Facts": 1,
    "Issue": 2,
    "Arguments of Petitioner": 3,
    "Arguments of Respondent": 4,
    "Reasoning": 5,
    "Decision": 6
}
label2rhetRole = {label: rhetRole for rhetRole, label in rhetRole2label.items()}

NUM_LABELS = len(rhetRole2label)
MAX_SEGMENT_LENGTH = 130
