import torch

from transformers import AutoTokenizer

from datasets import load_dataset

from helper import *
from model import *

# -----------------------------
# Example "main" skeleton
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_limit_bytes = 15 * 1024 ** 3
    print("Using device:", device)

    dsRaw = load_dataset("sentence-transformers/parallel-sentences-tatoeba", "en-es")

    # Tokenizer and vocab setup
    model_name = "roberta-base"
    tok = AutoTokenizer.from_pretrained(model_name)

    print(dsRaw["train"][0])

    pad_id = tok.pad_token_id
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    if pad_id is None:
        print("Not good" * 10)
        raise ValueError("No pad token found!")

    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing BOS/EOS token ids; cannot use for sequence generation.")

    SRC_VOCAB_SIZE = tok.vocab_size
    TGT_VOCAB_SIZE = tok.vocab_size

    specials = SpecialTokens(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

    def encode_src(text: str):
        return tok(text, add_special_tokens=True, truncation=True, max_length=128)["input_ids"]

    def encode_tgt(text: str):
        # do NOT add_special_tokens here; we add BOS/EOS ourselves
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
        return [specials.bos_id] + ids + [specials.eos_id]

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)

    # Model hyperparams
    model = Seq2SeqTransformer(
        num_encoder_layers=2, # 4
        num_decoder_layers=2, # 4 
        emb_size=256,         # 512
        nhead=4,              # 8
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim_feedforward=1024, # 2048
        dropout=0.1,
        pad_id=specials.pad_id,
    ).to(device)
    exit_if_vram_limit_exceeded(device, vram_limit_bytes, "model initialization")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.015)

    # Data Split
    dsTrain_size = dsRaw['train'].num_rows

    train_split = int(dsTrain_size * 0.8)

    # Training Data
    subset = dsRaw["train"].select(range(0, train_split))
    pairs = [(encode_src(ex["non_english"]), encode_tgt(ex["english"])) for ex in subset]

    ds = TranslationDataset(pairs, pad_id=specials.pad_id)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, specials.pad_id))

    # Validation Data
    valSet = dsRaw["train"].select(range(train_split, dsTrain_size))
    pairsVal = [(encode_src(ex["non_english"]), encode_tgt(ex["english"])) for ex in valSet]

    dsV = TranslationDataset(pairsVal, pad_id=specials.pad_id)
    dlV = DataLoader(dsV, batch_size=16, collate_fn=lambda b: collate_fn(b, specials.pad_id))

    # Call to train
    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(70):
        loss = train_one_epoch(
            model,
            dl,
            optimizer,
            device,
            specials,
            label_smoothing=0.1,
            vram_limit_bytes=vram_limit_bytes,
        )
        val_loss = evaluate(
            model,
            dlV,
            device,
            specials,
            label_smoothing=0.0,
            vram_limit_bytes=vram_limit_bytes,
        )
        train_losses.append(loss)
        val_losses.append(val_loss)
        print(f"epoch={epoch} loss={loss:.4f} val_loss={val_loss:.4f}")

    plot_learning_curve(train_losses, val_losses)

    # Inference on single example
    src_ids = encode_src("Yo soy un estudiante inteligente.")
    src_tensor = torch.tensor(src_ids).unsqueeze(0)  # (1, seq)
    decoded = greedy_decode(model, src_tensor, device, specials, max_len=30).squeeze(0).tolist()
    exit_if_vram_limit_exceeded(device, vram_limit_bytes, "greedy decoding")
    print("decoded ids", decoded)
    print("decoded example:", decode(decoded))

    # Test
    tSet = dsRaw["dev"]
    pairsT = [(encode_src(ex["non_english"]), encode_tgt(ex["english"])) for ex in tSet]

    dsT = TranslationDataset(pairsT, pad_id=specials.pad_id)
    dlT = DataLoader(dsT, batch_size=16, collate_fn=lambda b: collate_fn(b, specials.pad_id))

    test_loss = evaluate(
        model,
        dlT,
        device,
        specials,
        label_smoothing=0.0,
        vram_limit_bytes=vram_limit_bytes,
    )

    print(f"Test loss: {test_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "transformer_translation.pt")


if __name__ == "__main__":
    main()
