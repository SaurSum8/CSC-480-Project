import torch
import torchvision.transforms.functional as f


def add_label_id(ex, label2id):
    ex["label_id"] = label2id[ex["label"]]
    return ex


def preprocess(img):
    # Ensure 3-channel input for the CNN.
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
    # Normalize to speed up/stabilize optimization.
    img = f.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img


def process(batch):
    batch["pixel_values"] = [preprocess(im) for im in batch["image"]]
    return batch


def collate_fn(data):
    x = torch.stack([d["pixel_values"] for d in data])
    y = torch.tensor([d["label_id"] for d in data], dtype=torch.long)
    texts = [d["label"] for d in data]
    return {"pixel_values": x, "label_id": y, "labels": texts}


def format_param_size(num_bytes):
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"
