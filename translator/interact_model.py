import torch
from transformers import AutoTokenizer

from transformer import Seq2SeqTransformer, SpecialTokens, greedy_decode 

# IMPORTANT : CHANGE AS NECESSARY TO MATCH TRAINING HYPERPARAMS EXACTLY, AND ALSO TOKENS

def load_trained_model(
    weights_path: str = "transformer_translation.pt",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_name)

    specials = SpecialTokens(
        pad_id=tok.pad_token_id,
        bos_id=tok.cls_token_id,
        eos_id=tok.sep_token_id,
    )

    # must match training hyperparams exactly
    model = Seq2SeqTransformer(
        num_encoder_layers=4,
        num_decoder_layers=4,
        emb_size=512,
        nhead=8,
        src_vocab_size=tok.vocab_size,
        tgt_vocab_size=tok.vocab_size,
        dim_feedforward=2048,
        dropout=0.1,
        pad_id=specials.pad_id,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tok, specials, device


if __name__ == "__main__":
    model, tok, specials, device = load_trained_model()

    def encode_src(text: str):
        return tok(text, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)

    # Example
    src_ids = encode_src("Yo soy un estudiante inteligente.")
    src_tensor = torch.tensor(src_ids).unsqueeze(1)  # (seq, 1)
    decoded = greedy_decode(model, src_tensor, device, specials, max_len=30).squeeze(1).tolist()
    print("decoded ids", decoded)
    print("decoded example:", decode(decoded))

    # User Input Based Translation
    ans = ''
    while ans != 'exit':
        ans = input("Enter a Spanish sentence (or 'exit' to quit): ")
        if ans.lower() == 'exit':
            break
        src_ids = encode_src(ans)
        src_tensor = torch.tensor(src_ids).unsqueeze(1)  # (seq, 1)
        decoded = greedy_decode(model, src_tensor, device, specials, max_len=30).squeeze(1).tolist()
        print("Translation:", decode(decoded))

    # quick sanity check: number of parameters
    print("Loaded model on", device, "params:", sum(p.numel() for p in model.parameters()))