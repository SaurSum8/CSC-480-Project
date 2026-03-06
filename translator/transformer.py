import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Positional Encoding (classic)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) for seq-first batching
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# -----------------------------
# Seq2Seq Transformer Model
# -----------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.emb_size = emb_size

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we use (seq, batch, feature)
            norm_first=False,
        )

        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_id)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        src: (src_seq_len, batch)
        tgt: (tgt_seq_len, batch)
        masks: per nn.Transformer docs
        returns logits: (tgt_seq_len, batch, tgt_vocab_size)
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


# -----------------------------
# Mask helpers
# -----------------------------
def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Causal mask for target: (sz, sz) with -inf above diagonal.
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    out = torch.zeros(sz, sz, device=device)
    out = out.masked_fill(mask, float("-inf"))
    return out

def create_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    tokens: (seq_len, batch)
    returns: (batch, seq_len) bool mask (True = pad positions)
    """
    return (tokens.transpose(0, 1) == pad_id)


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
    Returns src, tgt as seq-first tensors: (seq_len, batch)
    """
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = [x + [pad_id] * (max_src - len(x)) for x in src_batch]
    tgt_padded = [x + [pad_id] * (max_tgt - len(x)) for x in tgt_batch]

    src = torch.tensor(src_padded, dtype=torch.long).transpose(0, 1)  # (src_seq, batch)
    tgt = torch.tensor(tgt_padded, dtype=torch.long).transpose(0, 1)  # (tgt_seq, batch)
    return src, tgt


# -----------------------------
# Training + decoding utilities
# -----------------------------
@dataclass
class SpecialTokens:
    pad_id: int
    bos_id: int
    eos_id: int

def train_one_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    specials: SpecialTokens,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # input to decoder is everything except last token
        # target labels are everything except first token
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask = torch.zeros((src.size(0), src.size(0)), device=device)  # no causal mask in encoder
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0), device=device)

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
        )  # (tgt_seq-1, batch, vocab)

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
    src: (src_seq_len, 1) single example
    returns: (decoded_len, 1)
    """
    model.eval()
    src = src.to(device)

    src_mask = torch.zeros((src.size(0), src.size(0)), device=device)
    src_padding_mask = create_padding_mask(src, specials.pad_id)

    memory = model.encode(src, src_mask, src_padding_mask)

    ys = torch.tensor([[specials.bos_id]], dtype=torch.long, device=device)  # (1,1)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device=device)
        tgt_padding_mask = create_padding_mask(ys, specials.pad_id)

        out = model.decode(
            tgt=ys,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = model.generator(out)  # (tgt_len, 1, vocab)
        next_token = torch.argmax(logits[-1, 0, :], dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=0)
        if next_token == specials.eos_id:
            break

    return ys


# -----------------------------
# Example "main" skeleton
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You MUST set these from your tokenizers/vocabs
    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000
    specials = SpecialTokens(pad_id=0, bos_id=1, eos_id=2)

    # Model hyperparams
    model = Seq2SeqTransformer(
        num_encoder_layers=4,
        num_decoder_layers=4,
        emb_size=512,
        nhead=8,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim_feedforward=2048,
        dropout=0.1,
        pad_id=specials.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Example toy data: replace with real tokenized sentence pairs
    # Each tgt sequence should be: [BOS] ... [EOS]
    pairs = [
        ([5, 10, 11, 12, 2], [1, 7, 8, 9, 2]),
        ([6, 13, 14, 2],    [1, 15, 16, 2]),
    ]
    ds = TranslationDataset(pairs, pad_id=specials.pad_id)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, specials.pad_id))

    for epoch in range(5):
        loss = train_one_epoch(model, dl, optimizer, device, specials, label_smoothing=0.1)
        print(f"epoch={epoch} loss={loss:.4f}")

    # Inference on single example
    src_example, _ = ds[0]
    src_tensor = torch.tensor(src_example, dtype=torch.long).unsqueeze(1)  # (seq, 1)
    decoded = greedy_decode(model, src_tensor, device, specials, max_len=30).squeeze(1).tolist()
    print("decoded token ids:", decoded)


if __name__ == "__main__":
    main()
