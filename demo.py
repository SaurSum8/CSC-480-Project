import string
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from CNN import helper as cnn_helper
from CNN.model import CNN
from transformer.helper import SpecialTokens, greedy_decode
from transformer.model import Seq2SeqTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CNN_WEIGHTS = BASE_DIR / "CNN" / "CNN_translation.pt"
TRANSFORMER_WEIGHTS = BASE_DIR / "transformer" / "transformer_translation.pt"


def build_cnn_vocab():
    base_charset = (
        string.digits
        + string.ascii_lowercase
        + string.ascii_uppercase
        + string.punctuation
        + " "
    )
    char2id = {ch: i + 1 for i, ch in enumerate(base_charset)}
    unk_id = len(char2id) + 1
    char2id["<unk>"] = unk_id
    id2char = {idx: ch for ch, idx in char2id.items() if ch != "<unk>"}
    id2char[unk_id] = "?"
    blank_id = 0
    num_classes = unk_id + 1
    return blank_id, id2char, num_classes


def load_cnn_model(device: torch.device):
    blank_id, id2char, num_classes = build_cnn_vocab()
    model = CNN(in_channels=3, num_classes=num_classes).to(device)
    state_dict = torch.load(CNN_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, blank_id, id2char


def load_transformer_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing pad/bos/eos ids needed for inference.")

    specials = SpecialTokens(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    model = Seq2SeqTransformer(
        num_encoder_layers=2,
        num_decoder_layers=2,
        emb_size=256,
        nhead=4,
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=specials.pad_id,
    ).to(device)

    state_dict = torch.load(TRANSFORMER_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, specials


@torch.no_grad()
def ocr_image(image_path: Path, cnn_model: CNN, device: torch.device, blank_id: int, id2char: dict[int, str]) -> str:
    with Image.open(image_path) as image:
        pixel_values = cnn_helper.preprocess(image).unsqueeze(0).to(device)

    logits = cnn_model(pixel_values)
    pred_ids = logits.argmax(dim=2)[0].detach().cpu().tolist()
    return cnn_helper.ctc_greedy_decode(pred_ids, id2char=id2char, blank_id=blank_id)


@torch.no_grad()
def translate_text(
    text: str,
    transformer_model: Seq2SeqTransformer,
    tokenizer: AutoTokenizer,
    device: torch.device,
    specials: SpecialTokens,
) -> str:
    src_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
    )["input_ids"]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    decoded_ids = greedy_decode(
        transformer_model,
        src_tensor,
        device,
        specials,
        max_len=128,
    ).squeeze(0).tolist()
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(DATA_DIR.rglob("*.webp"))

    if not image_paths:
        print(f"No .webp files found under {DATA_DIR}")
        return

    if not CNN_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing CNN weights at {CNN_WEIGHTS}")
    if not TRANSFORMER_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing transformer weights at {TRANSFORMER_WEIGHTS}")

    cnn_model, blank_id, id2char = load_cnn_model(device)
    transformer_model, tokenizer, specials = load_transformer_model(device)

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} .webp files")

    for image_path in image_paths:
        ocr_text = ocr_image(image_path, cnn_model, device, blank_id, id2char)
        translated_text = translate_text(
            ocr_text,
            transformer_model,
            tokenizer,
            device,
            specials,
        )

        print(f"\nFile: {image_path}")
        print(f"OCR: {ocr_text}")
        print(f"Transformer: {translated_text}")


if __name__ == "__main__":
    main()
