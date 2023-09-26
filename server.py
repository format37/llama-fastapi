from fastapi import FastAPI, Query
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict

app = FastAPI()

# simple tokenization by characters
def encode(s, stoi):
    return [stoi[ch] for ch in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

def generate(device, model, config, itos, max_new_tokens=30):
    idx = torch.zeros(5, 1).long().to(device)  # Move idx to the device
    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[
            :, -1, :
        ]  # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
        idx_next = torch.multinomial(
            p, num_samples=1
        )  # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
    return [decode(x, itos) for x in idx.tolist()]

def generate_from_text(device, model, config, itos, stoi, start_text, max_new_tokens=30):
    # Encode the starting text
    idx = torch.tensor([encode(start_text, stoi)], dtype=torch.long).to(device)

    # Generate tokens
    for _ in range(max_new_tokens):
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[:, -1, :]
        p = F.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    # Decode to get the generated text
    generated_text = decode(idx[0].tolist(), itos)
    return generated_text

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """
    def __init__(self, size, config):
        super().__init__()
        self.config = config
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    return R

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        self.R = self.R.to(x.device)  # Move R to the same device as x
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model'], config),
        )

    def forward(self, x):
        x = self.rms(x) # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.feedforward(x)
        return x

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )

        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model'], config),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits
        
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

# Train part:

def train_llama(device, model, optimizer, dataset, scheduler=None, config=None, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'], config)
        xs, ys = xs.to(device), ys.to(device)  # Move data to device
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(device, model, dataset, config)
            losses += [x]
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()

def get_batches(data, split, batch_size, context_window, config):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]
    
    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test
    
    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y


@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(device, model, dataset, config):
    out = {}
    model.eval()
    model.to(device)  # Ensure model is on the correct device
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'], config)
            xb, yb = xb.to(device), yb.to(device)  # Move data to the same device as model
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

@app.get("/")
def read_root():
    return {"message": "Welcome to the Llama API"}

@app.get("/generate/")
def generate(start_text: str = Query(..., min_length=1)):
    # Record the start time
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device:', device)

    lines = open('./data/input.txt', 'r').read()
    vocab = sorted(list(set(lines)))
    itos = {i:ch for i, ch in enumerate(vocab)}
    stoi = {ch:i for i, ch in enumerate(vocab)}
    vocab = sorted(list(set(lines)))

    MASTER_CONFIG = {
        'context_window': 16,
        "vocab_size": len(vocab),
        'd_model': 128,
        'epochs': 300,
        'log_interval': 100,
        'batch_size': 32,
        'n_layers': 4,
        'n_heads': 8,
    }

    config = {
        'batch_size': 32,
        'context_window': 11,
        'd_model': 13,
    }

    job_type_train = True

    dataset = torch.tensor(encode(lines, stoi), dtype=torch.int8)

    llama = Llama(MASTER_CONFIG).to(device)
    optimizer = torch.optim.Adam(llama.parameters())

    """if job_type_train:
        print('training llama')
        train(device, llama, optimizer, dataset, config=MASTER_CONFIG, print_logs=True)

        print('Saving the model')
        torch.save(llama.state_dict(), "llama.pt")
    else:
        print('Loading the model')
        llama.load_state_dict(torch.load("llama.pt"))"""
    print('Loading the model')
    llama.load_state_dict(torch.load("./data/llama.pt"))

    xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], config)
    xs, ys = xs.to(device), ys.to(device)  # Move data to device

    logits, loss = llama(xs, ys)

    print(loss)

    # Generate part:
    # print(generate(device, llama, MASTER_CONFIG, itos, 500)[0])
    generated_text = generate_from_text(device, llama, MASTER_CONFIG, itos, stoi, start_text=start_text, max_new_tokens=50)
    print(generated_text)

    # Record the end time
    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the time difference
    time_difference = end_time - start_time
    print(f"Time Difference: {str(time_difference)}")

    # generated_text = generate_text(start_text)
    return {"generated_text": generated_text}


@app.get("/train/")
def train():
    # Record the start time
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device:', device)

    lines = open('./data/input.txt', 'r').read()
    vocab = sorted(list(set(lines)))
    itos = {i:ch for i, ch in enumerate(vocab)}
    stoi = {ch:i for i, ch in enumerate(vocab)}
    vocab = sorted(list(set(lines)))

    MASTER_CONFIG = {
        'context_window': 16,
        "vocab_size": len(vocab),
        'd_model': 128,
        'epochs': 300,
        'log_interval': 30000,
        'batch_size': 32,
        'n_layers': 4,
        'n_heads': 8,
    }

    config = {
        'batch_size': 32,
        'context_window': 11,
        'd_model': 13,
    }

    job_type_train = True

    dataset = torch.tensor(encode(lines, stoi), dtype=torch.int8)

    llama = Llama(MASTER_CONFIG).to(device)
    optimizer = torch.optim.Adam(llama.parameters())

    print('training llama')
    train_llama(device, llama, optimizer, dataset, config=MASTER_CONFIG, print_logs=True)

    print('Saving the model')
    torch.save(llama.state_dict(), "data/llama.pt")

    xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], config)
    xs, ys = xs.to(device), ys.to(device)  # Move data to device

    logits, loss = llama(xs, ys)

    print(loss)

    # Generate part:
    # print(generate(device, llama, MASTER_CONFIG, itos, 500)[0])
    # generated_text = generate_from_text(device, llama, MASTER_CONFIG, itos, stoi, start_text=start_text, max_new_tokens=50)
    # print(generated_text)

    # Record the end time
    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the time difference
    time_difference = end_time - start_time
    print(f"Time Difference: {str(time_difference)}")

    # generated_text = generate_text(start_text)
    # return {"generated_text": generated_text}
    return {"loss": loss}
