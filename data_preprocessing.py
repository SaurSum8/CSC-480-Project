import os
import string
import torch
from datasets import load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader

import helper

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data Preprocessing
ds = load_dataset("priyank-m/MJSynth_text_recognition")

base_charset = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + " "
char2id = {ch: i + 1 for i, ch in enumerate(base_charset)}
unk_id = len(char2id) + 1
char2id["<unk>"] = unk_id
id2char = {idx: ch for ch, idx in char2id.items() if ch != "<unk>"}
id2char[unk_id] = "?"
blank_id = 0
num_classes = unk_id + 1

train_set = ds["train"].with_transform(helper.process)
test_set = ds["test"].with_transform(helper.process)
val_set = ds["val"].with_transform(helper.process)

num_workers = min(8, os.cpu_count() or 1)
loader_kwargs = {
    "batch_size": 64,
    "collate_fn": helper.build_collate_fn(char2id),
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
}
if num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_dataloader = DataLoader(train_set, shuffle=True, **loader_kwargs)
test_dataloader = DataLoader(test_set, shuffle=False, **loader_kwargs)
val_dataloader = DataLoader(val_set, shuffle=False, **loader_kwargs)
