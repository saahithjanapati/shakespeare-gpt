import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import wandb
import random

BLOCK_SIZE = 64
N_EMBD = 128
N_LAYERS = 6
N_HEAD = 8
DROPOUT_PROB = 0.2
USE_BIAS = False


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
def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


device = get_best_device()




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




############################################################################
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
        return len(self.data) - BLOCK_SIZE - 1

############################################################################

class MLP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD, bias=USE_BIAS)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD, bias=USE_BIAS)
        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

############################################################################

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=USE_BIAS)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=USE_BIAS)

        # (1, 1, BLOCK_SIZE, BLO)
        self.register_buffer("att_mask", torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE) * -float('inf'), diagonal=1).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

        self.attn_dropout = nn.Dropout(p=DROPOUT_PROB) # applied before c_proj, after softmax
        self.resid_dropout = nn.Dropout(p=DROPOUT_PROB) # applied right before returning


    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(split_size = N_EMBD, dim=2) # (B, T, C) --> (B, T, 3C) --> [(B,T,C), (B,T,C), (B,T,C)]
        
        # reshape the q,k,v tensors
        q = q.view(B, T, N_HEAD, -1).transpose(1,2) # (B,T,C) --> (B, T, N_H, H_DIM) --> (B, N_H, T, H_DIM)
        k = k.view(B, T, N_HEAD, -1).transpose(1,2) # (B,T,C) --> (B, T, N_H, H_DIM) --> (B, N_H, T, H_DIM)
        v = v.view(B, T, N_HEAD, -1).transpose(1,2) # (B,T,C) --> (B, T, N_H, H_DIM) --> (B, N_H, T, H_DIM)

        # (B, N_H, T, H_DIM) --> (B, N_H, H_DIM, T)
        att_scores =  q @ k.transpose(2, 3) / math.sqrt(C // N_HEAD) #(B, N_H, T, T)
        att_scores = att_scores + self.att_mask[:, :, :T, :T] # apply causal attention masking #(B, N_H, T, T)
        att_scores = torch.softmax(att_scores, dim=-1)

        att_scores = self.attn_dropout(att_scores)

        att_out  = att_scores @ v # (B, N_H, T, T) @ (B, N_H, T, H_DIM) --> (B, N_H, T, H_DIM)
        att_out = att_out.transpose(1,2).contiguous()
        att_out = att_out.view(B, T, C)

        out = self.c_proj(att_out)
        out = self.resid_dropout(out)
        return out

############################################################################

class Block(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape = (N_EMBD))
        self.sa = SelfAttention()
        self.ln_2 = nn.LayerNorm(normalized_shape = (N_EMBD))
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

######################################

class GPT(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)

        self.h = nn.ModuleList([Block() for _ in range(N_LAYERS)])

        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)


    def forward(self, inp, labels=None, ignore_index=-1):
        B, T = inp.shape
        tok_emb = self.tok_emb(inp) # (B,T) 
        pos_idx = torch.arange(T, device=next(self.parameters()).device)
        pos_emb = self.pos_emb(pos_idx) #

        x = tok_emb + pos_emb
        for layer_idx in range(N_LAYERS):
            x = self.h[layer_idx](x)

        logits =  self.lm_head(self.ln_f(x)) # (B, T, V)
        # labels should have shape (B, T)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=ignore_index)

        # compute loss if labels were provided
        return logits, loss

######################################
# sampling methods

def pick_next_greedy(model, inp):
    if inp.device != device:
        inp = inp.to(device)

    logits, _ = model(inp) # logits will have shape: (B, T, V)
    last_tok_logits = logits[:, -1, :] # (B, V)

    _, top_idx = torch.max(last_tok_logits, dim=1, keepdim=True)
    return top_idx



def pick_next_top_p(model, inp, p):
    """
    model chooses from the smallest possible set of tokens whose cumulative probability mass exceeds p
    """
    if inp.device != device:
        inp = inp.to(device)
        
    logits, _ = model(inp) # logits will have shape: (B, T, V)
    last_tok_logits = logits[:, -1, :] # (B, V)
    last_tok_probs = torch.softmax(last_tok_logits, dim=1) # (B, V)
    sorted_probs, sorted_indices = torch.sort(last_tok_probs, dim=-1, descending=True)

    # say p = 0.75
    # 0.5, 0.20,   0.2,  0.05,  0.05
    # 0.5, 0.70,  0.90,  0.95,  1.00
    # 0,   0.5   0.7,   

    cumsum = torch.cumsum(sorted_probs, dim=1)
    masked_probs = torch.where(cumsum - sorted_probs <= p, sorted_probs, 0) # (B, V)
    masked_probs = masked_probs / torch.sum(masked_probs, dim=1, keepdim=True)

    indices = torch.multinomial(masked_probs, 1) # (B, 1)
    actual_indices = torch.gather(sorted_indices, 1, indices)
    return actual_indices


def pick_next_top_k(model, inp, k):
    if inp.device != device:
        inp = inp.to(device)
        
    logits, _ = model(inp) # logits will have shape: (B, T, V)
    last_tok_logits = logits[:, -1, :] # (B, V)
    topk_vals, _ = torch.topk(last_tok_logits, k=k, dim=1) # (B, k)
    kth_val = topk_vals[:, [-1]] # (B, 1)

    last_tok_logits = torch.where(last_tok_logits >= kth_val, last_tok_logits, -float('inf')) # (B, V)
    last_tok_probs = torch.softmax(last_tok_logits, dim=1) # (B, V)

    next_tokens = torch.multinomial(last_tok_probs, 1) # (B, 1)
    return next_tokens


def generate(model, start_prompt, top_p=None, top_k=None, num_tokens=50, num_samples=16):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        start_tokens = []
        # tokenize the start prompt
        for c in start_prompt:
            if c not in stoi:
                raise ValueError(f"{c} was found in start_prompt for generate, but is not present in vocabulary")
            start_tokens.append(stoi[c])

        inp = torch.tensor(start_tokens, dtype=torch.long)
        inp = inp.to(next(model.parameters()).device)
        inp = inp.view(1, -1).expand(num_samples, inp.size(0))

        for i in range(num_tokens):
            if inp.size(1) > BLOCK_SIZE:
                x = inp[:, -BLOCK_SIZE:]
            else:
                x = inp

            if top_p:
                next_tokens = pick_next_top_p(model, x, top_p)
            elif top_k:
                next_tokens = pick_next_top_k(model, x, top_k)
            else:
                next_tokens = pick_next_greedy(model, x)

            # concatenate with inp
            inp = torch.cat([inp, next_tokens], dim=1)
        
        # move back to cpu, decode back to strings and return
        generations = []
        inp = inp.cpu()
        for i in range(len(inp)):
            curr_generation = "".join([itos[i] for i in inp[i, :].tolist()])
            generations.append(curr_generation)

    if was_training:
        model.train()
    return generations

##########################################

# training time
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
learning_rate = 1e-4

num_epochs = 2
eval_every = 500
print_every = 100
sample_prefix = "WHEREFORE ARE THOU ROMEO"
sample_every = 500
num_sample_tokens = 200
top_k = 13; top_p = None

# define datasets
train_dataset = ShakespeareDataset(split='train')
val_dataset = ShakespeareDataset(split='val')
# test_dataset = ShakespeareDataset(split='test')

# define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size = EVAL_BATCH_SIZE, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size = EVAL_BATCH_SIZE, shuffle=False)
def run_eval():
    model.eval()
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, labels=y)
            total_loss += loss.item() * len(x)
            num_samples +=  len(x)
    model.train() # put back into training mode
    return total_loss / num_samples



# define model and optimzier
model = GPT()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)



run_name = "exp_1"
wandb.init(
    project="shakespeare-gpt",
    name=run_name,
    config={
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_layers": N_LAYERS,
        "n_head": N_HEAD,
        "dropout": DROPOUT_PROB,
        "use_bias": USE_BIAS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "eval_every": eval_every,
        "print_every": print_every,
    },
)
wandb.watch(model, log="gradients", log_freq=print_every)

iter_idx = 0
total_num_iter = num_epochs * len(train_dataloader)
# core training loop
for epoch in range(num_epochs):

    model.train()
    for x,y in train_dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % print_every == 0:
            train_loss = loss.item()
            print(f"Iter - {iter_idx}/{total_num_iter}, Train Loss: {train_loss}")
            wandb.log({"train/loss": train_loss, "epoch": epoch, "iter": iter_idx}, step=iter_idx)
        
        if iter_idx % eval_every == 0:
            eval_loss = run_eval()
            print(f"Iter - {iter_idx}/{total_num_iter}, Val Loss: {eval_loss}")
            wandb.log({"val/loss": eval_loss, "epoch": epoch, "iter": iter_idx}, step=iter_idx)

        if iter_idx % sample_every == 0:
            generations = generate(
                model,
                start_prompt=sample_prefix,
                top_p=top_p,
                top_k=top_k,
                num_tokens=num_sample_tokens,
                num_samples=1,
            )
            if generations:
                sample_text = generations[0]
                print(f"Iter - {iter_idx}/{total_num_iter}, Sample Generation:\n{sample_text}")
                wandb.log({"sample/text": sample_text, "epoch": epoch, "iter": iter_idx}, step=iter_idx)


        iter_idx += 1


# save the model state at the end so we can evaluate later (we don't really need the optimizer, but just in case)
model_path = SCRIPT_DIR / "model.pth"
torch.save(model.state_dict(), model_path)
wandb.save(str(model_path))
wandb.finish()
