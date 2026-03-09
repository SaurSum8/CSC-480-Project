import torch
import torchvision.transforms.functional as f


def preprocess(img):
    # ensure 3-channel input for the CNN.
    if getattr(img, "mode", None) != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    new_w = max(1, int(w * (32 / h)))
    img = f.resize(img, (32, new_w))

    if new_w < 256:
        img = f.pad(img, [0, 0, 256 - new_w, 0], fill=0)
    else:
        img = f.crop(img, 0, 0, 32, 256)

    img = f.to_tensor(img)
    # normalize to speed up/stabilize optimization.
    img = f.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img


def process(batch):
    batch["pixel_values"] = [preprocess(im) for im in batch["image"]]
    return batch


def build_collate_fn(char2id):
    unk_id = char2id["<unk>"]

    def collate_fn(data):
        x = torch.stack([d["pixel_values"] for d in data])
        texts = [d["label"] for d in data]

        encoded = [
            torch.tensor([char2id.get(ch, unk_id) for ch in text], dtype=torch.long)
            for text in texts
        ]
        target_lengths = torch.tensor([seq.numel() for seq in encoded], dtype=torch.long)

        non_empty = [seq for seq in encoded if seq.numel() > 0]
        if non_empty:
            targets = torch.cat(non_empty)
        else:
            targets = torch.empty(0, dtype=torch.long)

        return {
            "pixel_values": x,
            "targets": targets,
            "target_lengths": target_lengths,
            "labels": texts,
        }

    return collate_fn


def ctc_greedy_decode(token_ids, id2char, blank_id=0):
    decoded = []
    prev = blank_id
    for idx in token_ids:
        if idx != blank_id and idx != prev:
            decoded.append(id2char.get(idx, ""))
        prev = idx
    return "".join(decoded)


def format_param_size(num_bytes):
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"
