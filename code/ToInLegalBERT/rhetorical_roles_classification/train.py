import os

import numpy as np
import optuna
import torch
import transformers
import wandb
from tqdm import tqdm

from rhetorical_roles_classification import MetricsTracker


def train(
    model,
    train_dataset,
    valid_dataset,
    lr=5e-5,
    weight_decay=0.01,
    num_warmup_steps=0,
    batch_size=1,
    epochs=10,
    accum_iter=3,
    early_stopping=True,
    early_stopping_patience=2,
    device="cuda",
    optuna_trial=None,
    use_wandb=False,
    destination_path=None,
):
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    model.to(device)
    metrics_tracker = MetricsTracker(device=device)
    optimizer = get_optimizer(model=model, weight_decay=weight_decay, lr=lr)
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    scaler = torch.cuda.amp.GradScaler()

    print("\n************** Training Started **************\n")

    if early_stopping:
        best_valid_loss = np.infty
        not_improving_epochs = 0
        stop = False

    for epoch in range(1, epochs + 1):
        print(f"EPOCH N. {epoch}")

        # --------
        # TRAINING
        # --------

        model.train()
        metrics_tracker.reset()
        train_loss = 0

        for batch_idx, (data, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", leave=True):
            with torch.cuda.amp.autocast():
                labels = labels.to(device)
                output = model(data.to(device), labels=labels)
                loss, logits = output.loss, output.logits
                predictions = logits.argmax(dim=-1)
                metrics_tracker.accumulate(predictions, labels)
                train_loss += loss.item()
                loss /= accum_iter

            scaler.scale(loss).backward()

            if not (batch_idx + 1) % accum_iter or batch_idx + 1 == len(train_dataloader):
                scaler.unscale_(optimizer)  # To clip unscaled gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        print("\nTRAIN RESULTS")
        train_metrics = metrics_tracker.get()
        train_loss /= len(train_dataset)
        print(f"Loss: {train_loss}")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value}")

        # ----------
        # EVALUATION
        # ----------

        model.eval()
        metrics_tracker.reset()
        valid_loss = 0

        with torch.no_grad():
            for data, labels in valid_dataloader:
                labels = labels.to(device)
                output = model(data.to(device), labels=labels)
                loss, logits = output.loss, output.logits
                predictions = logits.argmax(dim=-1)
                metrics_tracker.accumulate(predictions, labels)
                valid_loss += loss

        print("\nVALIDATION RESULTS")
        valid_metrics = metrics_tracker.get()
        valid_loss /= len(valid_dataset)
        print(f"Loss: {valid_loss}")
        for metric, value in valid_metrics.items():
            print(f"{metric}: {value}")

        if early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                not_improving_epochs = 0
                if destination_path is not None:
                    torch.save(model.state_dict(), os.path.join(destination_path))
            else:
                not_improving_epochs += 1
                if not_improving_epochs == early_stopping_patience:
                    print("Early stopped training")
                    stop = True

        print("--------------------------------------------------")

        if use_wandb:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Train Metrics": train_metrics,
                    "Validation Loss": valid_loss,
                    "Valid Metrics": valid_metrics,
                }
            )

        if optuna_trial is not None:
            optuna_trial.report(valid_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopping and stop:
            break

    return best_valid_loss


def get_optimizer(model, weight_decay, lr):
    params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(grouped_params, lr=lr)


def get_scheduler(
    optimizer,
    num_training_steps,
    num_warmup_steps,
):
    if isinstance(num_warmup_steps, float):
        num_warmup_steps = int(num_training_steps * num_warmup_steps)
    else:
        num_warmup_steps = num_warmup_steps

    return transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
