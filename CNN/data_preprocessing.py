import os
import string
import torch
from datasets import load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader

from CNN.helper import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data Preprocessing
ds = load_dataset("priyank-m/MJSynth_text_recognition")


def maybe_limit_split(split, env_name: str):
    limit = os.getenv(env_name)
    if not limit:
        return split

    limit = int(limit)
    return split.select(range(min(limit, split.num_rows)))

base_charset = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + " "
char2id = {ch: i + 1 for i, ch in enumerate(base_charset)}
unk_id = len(char2id) + 1
char2id["<unk>"] = unk_id
id2char = {idx: ch for ch, idx in char2id.items() if ch != "<unk>"}
id2char[unk_id] = "?"
blank_id = 0
num_classes = unk_id + 1 # 97

train_set = maybe_limit_split(ds["train"], "CNN_TRAIN_LIMIT").with_transform(process)
test_set = maybe_limit_split(ds["test"], "CNN_TEST_LIMIT").with_transform(process)
val_set = maybe_limit_split(ds["val"], "CNN_VAL_LIMIT").with_transform(process)

num_workers = min(int(os.getenv("CNN_NUM_WORKERS", "2")), os.cpu_count() or 1)
loader_kwargs = {
    "batch_size": int(os.getenv("CNN_BATCH_SIZE", "64")),
    "collate_fn": build_collate_fn(char2id),
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
}
if num_workers > 0 and os.getenv("CNN_PERSISTENT_WORKERS", "0") == "1":
    loader_kwargs["persistent_workers"] = True
if num_workers > 0 and os.getenv("CNN_PREFETCH_FACTOR"):
    loader_kwargs["prefetch_factor"] = int(os.getenv("CNN_PREFETCH_FACTOR"))

train_dataloader = DataLoader(train_set, shuffle=True, **loader_kwargs)
test_dataloader = DataLoader(test_set, shuffle=False, **loader_kwargs)
val_dataloader = DataLoader(val_set, shuffle=False, **loader_kwargs)
