import time
from datetime import datetime
import os
from pathlib import Path

import torch
from torch import nn

from CNN import helper
from CNN.model import CNN
from CNN.data_preprocessing import train_dataloader, val_dataloader, test_dataloader, blank_id, id2char, num_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, num_classes=num_classes).to(device)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.txt"
WEIGHTS_PATH = BASE_DIR / "CNN_translation.pt"

def log_line(message: str, echo: bool = True) -> None:
    if echo:
        print(message)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_header() -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Run started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Total params: {total_params:,} ({helper.format_param_size(total_params * 4)})\n")
        f.write(f"Trainable params: {trainable_params:,} ({helper.format_param_size(trainable_params * 4)})\n")
        f.write(f"Non-trainable params: {non_trainable_params:,} ({helper.format_param_size(non_trainable_params * 4)})\n")
        f.write("-" * 80 + "\n")


loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
num_epochs = int(os.getenv("CNN_EPOCHS", "15"))


def read_step_limit(env_name: str, default: int):
    value = int(os.getenv(env_name, str(default)))
    return None if value <= 0 else value


train_steps_per_epoch = read_step_limit("CNN_TRAIN_STEPS_PER_EPOCH", 3000)
val_steps_per_epoch = read_step_limit("CNN_VAL_STEPS_PER_EPOCH", 0)
test_steps = read_step_limit("CNN_TEST_STEPS", 0)

log_header()
log_line(
    "Training setup | "
    f"epochs={num_epochs} | "
    "optimizer=AdamW(lr=3e-4, weight_decay=1e-5) | "
    f"train_steps_per_epoch={train_steps_per_epoch or 'full'} | "
    f"val_steps_per_epoch={val_steps_per_epoch or 'full'} | "
    f"test_steps={test_steps or 'full'}",
    echo=False,
)


def run_epoch(dataloader, training=True, global_step=0, max_steps=None, compute_exact=None):
    if compute_exact is None:
        compute_exact = not training

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_exact_seen = 0

    for step_idx, batch in enumerate(dataloader, start=1):
        x = batch["pixel_values"].to(device)
        targets = batch["targets"].to(device=device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(device=device, dtype=torch.long)
        labels = batch["labels"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(x)
            log_probs = logits.log_softmax(dim=2).transpose(0, 1)
            input_lengths = torch.full(
                (x.size(0),), log_probs.size(0), dtype=torch.long, device=device
            )
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1

                if global_step % 100 == 0:
                    log_line(f"step={global_step} train_batch_loss={loss.item():.4f}")

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size

        if compute_exact:
            pred_ids = logits.argmax(dim=2).detach().cpu().tolist()
            pred_texts = [
                helper.ctc_greedy_decode(ids, id2char=id2char, blank_id=blank_id) for ids in pred_ids
            ]
            total_correct += sum(int(pred == label) for pred, label in zip(pred_texts, labels))
            total_exact_seen += batch_size

        if max_steps is not None and step_idx >= max_steps:
            break

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = None if not compute_exact else total_correct / max(total_exact_seen, 1)
    return avg_loss, avg_acc, global_step


run_start = time.perf_counter()
global_step = 0
for epoch in range(num_epochs):
    epoch_start = time.perf_counter()

    train_loss, train_acc, global_step = run_epoch(
        train_dataloader,
        training=True,
        global_step=global_step,
        max_steps=train_steps_per_epoch,
        compute_exact=False,
    )
    val_loss, val_acc, _ = run_epoch(
        val_dataloader,
        training=False,
        global_step=global_step,
        max_steps=val_steps_per_epoch,
        compute_exact=True,
    )

    elapsed_s = time.perf_counter() - epoch_start
    log_line(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"train_loss={train_loss:.4f} train_exact=n/a | "
        f"val_loss={val_loss:.4f} val_exact={val_acc:.4f} | "
        f"{elapsed_s:.1f}s"
    )

torch.save(model.state_dict(), WEIGHTS_PATH)

test_loss, test_acc, _ = run_epoch(
    test_dataloader,
    training=False,
    global_step=global_step,
    max_steps=test_steps,
    compute_exact=True,
)
log_line(f"Test | loss={test_loss:.4f} exact={test_acc:.4f}")

run_elapsed = time.perf_counter() - run_start
log_line(f"Run finished: {datetime.now().isoformat(timespec='seconds')} | total_time_s={run_elapsed:.1f}", echo=False)
