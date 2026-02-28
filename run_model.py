import time
import torch
from torch import nn

import helper
from model import CNN
from data_preprocessing import label2id, train_dataloader, val_dataloader, test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, num_classes=len(label2id)).to(device)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total params: {total_params:,} ({helper.format_param_size(total_params * 4)})")
print(f" Trainable params: {trainable_params:,} ({helper.format_param_size(trainable_params * 4)})")
print(f" Non-trainable params: {non_trainable_params:,} ({helper.format_param_size(non_trainable_params * 4)})")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
num_epochs = 15


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
        y = batch["label_id"].to(device=device, dtype=torch.long)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1

                if global_step % 100 == 0:
                    print(f"step={global_step} loss={loss.item():.4f}")

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += batch_size

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc, global_step


global_step = 0
for epoch in range(num_epochs):
    epoch_start = time.perf_counter()

    train_loss, train_acc, global_step = run_epoch(
        train_dataloader, training=True, global_step=global_step
    )
    val_loss, val_acc, _ = run_epoch(val_dataloader, training=False, global_step=global_step)

    elapsed_s = time.perf_counter() - epoch_start
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
        f"{elapsed_s:.1f}s"
    )


test_loss, test_acc, _ = run_epoch(test_dataloader, training=False, global_step=global_step)
print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")
