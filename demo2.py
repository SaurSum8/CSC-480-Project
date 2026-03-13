import string
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from CNN import helper as cnn_helper
from CNN.model import CNN

from local.transformer import Seq2SeqTransformer, SpecialTokens, greedy_decode 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CNN_WEIGHTS = BASE_DIR / "CNN" / "CNN_translation.pt"
TRANSFORMER_WEIGHTS = BASE_DIR / "transformer" / "transformer_translation4.pt"


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


def load_trained_model(
    weights_path: str = TRANSFORMER_WEIGHTS,
    model_name: str = "Helsinki-NLP/opus-mt-en-es",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_name)

    specials = SpecialTokens(
        pad_id=tok.pad_token_id,
        #bos_id=tok.cls_token_id,
        #eos_id=tok.sep_token_id,
        bos_id=tok.eos_token_id,
        eos_id=tok.eos_token_id,
    )

    # must match training hyperparams exactly
    model = Seq2SeqTransformer(
        num_encoder_layers=6,
        num_decoder_layers=6,
        emb_size=384,
        nhead=6,
        src_vocab_size=tok.vocab_size,
        tgt_vocab_size=tok.vocab_size,
        dim_feedforward=1536,
        dropout=0.2,
        pad_id=specials.pad_id,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tok, specials, device


@torch.no_grad()
def ocr_image(image_path: Path, cnn_model: CNN, device: torch.device, blank_id: int, id2char: dict[int, str]) -> str:
    with Image.open(image_path) as image:
        pixel_values = cnn_helper.preprocess(image).unsqueeze(0).to(device)

    logits = cnn_model(pixel_values)
    pred_ids = logits.argmax(dim=2)[0].detach().cpu().tolist()
    return cnn_helper.ctc_greedy_decode(pred_ids, id2char=id2char, blank_id=blank_id)


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
    model, tok, specials, device = load_trained_model()

    def encode_src(text: str):
        return tok(text, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} .webp files")

    sentence = ""

    for image_path in image_paths:
        ocr_text = ocr_image(image_path, cnn_model, device, blank_id, id2char)
        sentence += ocr_text + " "

        print(f"\nFile: {image_path}")
        print(f"OCR: {ocr_text}")

    src_ids = encode_src(sentence.strip())
    src_tensor = torch.tensor(src_ids).unsqueeze(1)  # (seq, 1)
    decoded = greedy_decode(model, src_tensor, device, specials, max_len=30).squeeze(1).tolist()
    
    print("Sentence to translate:", sentence.strip())
    print("Transformer translation:", decode(decoded))

if __name__ == "__main__":
    main()
