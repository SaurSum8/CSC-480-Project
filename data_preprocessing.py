import os
import torch
from datasets import load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader

import helper

# Allow PIL to load truncated files instead of throwing OSError.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data Preprocessing
ds = load_dataset("priyank-m/MJSynth_text_recognition")
labels = ds["train"].unique("label")
label2id = {lab: i for i, lab in enumerate(labels)}
ds = ds.map(helper.add_label_id, fn_kwargs={"label2id": label2id})

train_set = ds["train"].with_transform(helper.process)
test_set = ds["test"].with_transform(helper.process)
val_set = ds["val"].with_transform(helper.process)

num_workers = min(8, os.cpu_count() or 1)
loader_kwargs = {
    "batch_size": 64,
    "collate_fn": helper.collate_fn,
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
}
if num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_dataloader = DataLoader(train_set, shuffle=True, **loader_kwargs)
test_dataloader = DataLoader(test_set, shuffle=False, **loader_kwargs)
val_dataloader = DataLoader(val_set, shuffle=False, **loader_kwargs)
