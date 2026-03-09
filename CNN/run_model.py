import time
from datetime import datetime

import torch
from torch import nn

import helper
from model import CNN
from CNN.data_preprocessing import train_dataloader, val_dataloader, test_dataloader, blank_id, id2char, num_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, num_classes=num_classes).to(device)

RESULTS_PATH = "results.txt"

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
num_epochs = 15

log_header()
log_line(f"Training setup | epochs={num_epochs} | optimizer=AdamW(lr=3e-4, weight_decay=1e-5)", echo=False)


def run_epoch(dataloader, training=True, global_step=0):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in dataloader:
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

        pred_ids = logits.argmax(dim=2).detach().cpu().tolist()
        pred_texts = [
            helper.ctc_greedy_decode(ids, id2char=id2char, blank_id=blank_id) for ids in pred_ids
        ]
        total_correct += sum(int(pred == label) for pred, label in zip(pred_texts, labels))

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc, global_step


run_start = time.perf_counter()
global_step = 0
for epoch in range(num_epochs):
    epoch_start = time.perf_counter()

    train_loss, train_acc, global_step = run_epoch(
        train_dataloader, training=True, global_step=global_step
    )
    val_loss, val_acc, _ = run_epoch(val_dataloader, training=False, global_step=global_step)

    elapsed_s = time.perf_counter() - epoch_start
    log_line(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"train_loss={train_loss:.4f} train_exact={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_exact={val_acc:.4f} | "
        f"{elapsed_s:.1f}s"
    )


test_loss, test_acc, _ = run_epoch(test_dataloader, training=False, global_step=global_step)
log_line(f"Test | loss={test_loss:.4f} exact={test_acc:.4f}")

run_elapsed = time.perf_counter() - run_start
log_line(f"Run finished: {datetime.now().isoformat(timespec='seconds')} | total_time_s={run_elapsed:.1f}", echo=False)
