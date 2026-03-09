import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import torch.nn.functional as F

from transformer.model import *

# -----------------------------
# Training + decoding utilities
# -----------------------------
@dataclass
class SpecialTokens:
    pad_id: int
    bos_id: int
    eos_id: int

def exit_if_vram_limit_exceeded(device: torch.device, limit_bytes: Optional[int], context: str) -> None:
    if limit_bytes is None or device.type != "cuda" or not torch.cuda.is_available():
        return

    peak_reserved = torch.cuda.max_memory_reserved(device)
    if peak_reserved <= limit_bytes:
        return

    current_reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    limit_gb = limit_bytes / (1024 ** 3)
    print(
        f"VRAM limit exceeded during {context}: "
        f"current_reserved={current_reserved / (1024 ** 3):.2f} GiB, "
        f"peak_reserved={peak_reserved / (1024 ** 3):.2f} GiB, "
        f"peak_allocated={peak_allocated / (1024 ** 3):.2f} GiB, "
        f"limit={limit_gb:.2f} GiB. Exiting."
    )
    raise SystemExit(1)

def train_one_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    specials: SpecialTokens,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    vram_limit_bytes: Optional[int] = None,
):
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # input to decoder is everything except last token
        # target labels are everything except first token
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = None  # no causal mask in encoder
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

        src_padding_mask = create_padding_mask(src, specials.pad_id)
        tgt_padding_mask = create_padding_mask(tgt_input, specials.pad_id)
        memory_key_padding_mask = src_padding_mask

        logits = model(
            src=src,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (batch, tgt_seq-1, vocab)

        optimizer.zero_grad(set_to_none=True)

        # Flatten for CE loss
        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        tgt_out_flat = tgt_out.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            tgt_out_flat,
            ignore_index=specials.pad_id,
            label_smoothing=label_smoothing,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        exit_if_vram_limit_exceeded(device, vram_limit_bytes, "train_one_epoch")
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    device: torch.device,
    specials: SpecialTokens,
    label_smoothing: float = 0.0,
    vram_limit_bytes: Optional[int] = None,
):
    model.eval()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = None  # no causal mask in encoder
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

        src_padding_mask = create_padding_mask(src, specials.pad_id)
        tgt_padding_mask = create_padding_mask(tgt_input, specials.pad_id)
        memory_key_padding_mask = src_padding_mask

        logits = model(
            src=src,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (batch, tgt_seq-1, vocab)

        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        tgt_out_flat = tgt_out.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            tgt_out_flat,
            ignore_index=specials.pad_id,
            label_smoothing=label_smoothing,
        )

        exit_if_vram_limit_exceeded(device, vram_limit_bytes, "evaluate")
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    device: torch.device,
    specials: SpecialTokens,
    max_len: int = 60,
) -> torch.Tensor:
    """
    src: (1, src_seq_len) single example
    returns: (1, decoded_len)
    """
    model.eval()
    src = src.to(device)

    src_mask = None
    src_padding_mask = create_padding_mask(src, specials.pad_id)

    memory = model.encode(src, src_mask, src_padding_mask)

    ys = torch.tensor([[specials.bos_id]], dtype=torch.long, device=device)  # (1,1)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device)
        tgt_padding_mask = create_padding_mask(ys, specials.pad_id)

        out = model.decode(
            tgt=ys,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = model.generator(out)  # (1, tgt_len, vocab)
        next_token = torch.argmax(logits[0, -1, :], dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == specials.eos_id:
            break

    return ys

def plot_learning_curve(train_losses: List[float], val_losses: List[float], output_path: str = "result/learning_curve.png"):
    epochs = range(1, len(train_losses) + 1)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


# -----------------------------
# Minimal dataset + collate
# -----------------------------
class TranslationDataset(Dataset):
    """
    Expects pairs of (src_ids, tgt_ids) where each is a List[int].
    tgt_ids should include BOS ... EOS (or you can add them in collate_fn).
    """
    def __init__(self, pairs: List[Tuple[List[int], List[int]]], pad_id: int):
        self.pairs = pairs
        self.pad_id = pad_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch, pad_id: int):
    """
    Returns src, tgt as batch-first tensors: (batch, seq_len)
    """
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = [x + [pad_id] * (max_src - len(x)) for x in src_batch]
    tgt_padded = [x + [pad_id] * (max_tgt - len(x)) for x in tgt_batch]

    src = torch.tensor(src_padded, dtype=torch.long)  # (batch, src_seq)
    tgt = torch.tensor(tgt_padded, dtype=torch.long)  # (batch, tgt_seq)
    return src, tgt
