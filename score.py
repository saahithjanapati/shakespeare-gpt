import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import json

MAX_BLOCK_SIZE = 1024 # the max block size any of our models will have (defined here to ensure we evaluate on the exact same data splits)


###################### FILL THIS OUT WITH MODEL DETAILS ######################
from exp1.run import GPT # load the model class defined in the experiment file
path_to_experiment = Path("exp1/")
path_to_model = "exp1/model.pth"
BLOCK_SIZE = 64
############################################################################
# -----------------------------------------------------------------------------
# File-path helpers for local vs. Google Drive execution
# -----------------------------------------------------------------------------

IN_COLAB = "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ is not defined inside interactive/Colab notebooks
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent


def ensure_drive_mounted() -> Optional[Path]:
    """Mount Google Drive when running inside Colab and return the MyDrive path."""
    if not IN_COLAB:
        return None

    drive_root = Path("/content/drive")
    mydrive = drive_root / "MyDrive"
    if mydrive.exists():
        return mydrive

    try:
        from google.colab import drive as gdrive  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "google.colab module not available; cannot mount Google Drive."
        ) from exc

    gdrive.mount(str(drive_root))
    return mydrive


MYDRIVE_ROOT = ensure_drive_mounted()


def resolve_data_file(filename: str) -> Path:
    """
    Return the first existing path to `filename` among known data locations.

    Order:
    1. Original relative path (current working directory based)
    2. Repository structure relative to this script
    3. Google Drive repo mirror (if running inside Colab)
    """
    candidates = [
        Path("..") / "data" / filename,
        PROJECT_ROOT / "data" / filename,
        Path("data") / filename
    ]

    if MYDRIVE_ROOT:
        candidates.extend(
            [
                MYDRIVE_ROOT / "ai-laboratory" / "shakespeare-gpt" / "data" / filename,
                MYDRIVE_ROOT / "shakespeare-gpt" / "data" / filename,
                MYDRIVE_ROOT / "data" / filename,
            ]
        )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate {filename}. Checked: {', '.join(str(p) for p in candidates)}"
    )


def read_data_file(filename: str) -> str:
    data_path = resolve_data_file(filename)
    with data_path.open("r", encoding="utf-8") as handle:
        return handle.read()

# ------------------------------------------------------------
# read and tokenize the data at the character level
train_text = read_data_file("train.txt")
val_text = read_data_file("val.txt")
test_text = read_data_file("test.txt")


# note: this should be deterministic
combined_text = train_text + val_text + test_text
unique_chars = sorted(list(set(combined_text)))
VOCAB_SIZE = len(unique_chars)


stoi = {c: i for i,c in enumerate(unique_chars)}
itos = {i: c for i,c in enumerate(unique_chars)}


train_tokens = [stoi[c] for c in train_text]
val_tokens = [stoi[c] for c in val_text]
test_tokens = [stoi[c] for c in test_text]


class ShakespeareDataset(Dataset):
    def __init__(self, split):
        # fetch the tokens
        if split == 'train':
            data = train_tokens
        elif split == 'val':
            data = val_tokens
        elif split == 'test':
            data = test_tokens
        else:
            raise ValueError(f"split argument must be one of: [train, val, test], you gave {split}")

        self.data = torch.tensor(data)

    def __getitem__(self, idx):
        x = self.data[idx: idx + BLOCK_SIZE]
        y = self.data[idx+1: idx + 1 + BLOCK_SIZE]
        return x, y

    def __len__(self):
        return len(self.data) - MAX_BLOCK_SIZE - 1

############################################################################

def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


device = get_best_device()
# compute the loss on train, val, and test sets of the specified model
model = GPT()
model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
model.to(device)


# define datasets and dataloaders
train_dataset = ShakespeareDataset('train')
val_dataset = ShakespeareDataset('val')
test_dataset = ShakespeareDataset('test')

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=256)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=256)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=256)


def run_eval(model, dataloader):
    total_loss = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x,y = x.to(device), y.to(device)
            _, loss = model(x, labels=y)
            total_loss += loss.item() * len(x)
            num_samples += len(x)
    return total_loss / num_samples



print(f"Evaluating {path_to_model} on Training Set")
train_loss = run_eval(model, train_dataloader)

print(f"Evaluating {path_to_model} on Validation Set")
val_loss = run_eval(model, val_dataloader)

print(f"Evaluating {path_to_model} on Test Set")
test_loss = run_eval(model, test_dataloader)

path_to_results_file = path_to_experiment / "results.json"
results_json = {
    "train_loss": train_loss,
    "validation_loss": val_loss,
    "test_loss": test_loss,
    "wandb_run": ""
}

with open(path_to_results_file, "w") as f:
    json.dump(results_json, f, indent=2)


# write to file
# write file to experiment 
# update leaderboard