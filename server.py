from fastapi import FastAPI, Query, Body
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
import logging
import itertools
import os
from pydantic import BaseModel
import json

app = FastAPI()
conversation_history = []
# Create logger object
logger = logging.getLogger(__name__)
# Set the log level to DEBUG. This will print all the logs to the console
logger.setLevel(logging.DEBUG)
# Create console handler
ch = logging.StreamHandler()
# Set the log level to DEBUG. This will print all the logs to the console
ch.setLevel(logging.DEBUG)
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set formatter to console handler
ch.setFormatter(formatter)
# Add console handler to logger
logger.addHandler(ch)


# Chatbot part ++
def get_dialogue_batches(data, batch_size, stoi):
    random_indices = np.random.randint(0, len(data), batch_size)
    batch_dialogues = [data[i] for i in random_indices]
    # print("Dialogues before encoding: ", batch_dialogues)
    # logger.debug("Dialogues before encoding: %s", batch_dialogues)
    
    xs, ys = [], []
    for dialogue in batch_dialogues:
        user_text = dialogue.get('user')
        bot_text = dialogue.get('bot')
        encoded_xs = encode_dialogue({"user": user_text}, stoi)
        encoded_ys = encode_dialogue({"bot": bot_text}, stoi)
        
        if encoded_xs is not None:
            xs.append(encoded_xs)
        else:
            # print(f"Encoded xs is None for dialogue: {dialogue}")
            logger.debug("Encoded xs is None for dialogue: %s", dialogue)
        if encoded_ys is not None:
            ys.append(encoded_ys)
        else:
            # print(f"Encoded ys is None for dialogue: {dialogue}")
            logger.debug("Encoded ys is None for dialogue: %s", dialogue)
        # print("Length of xs: ", len(xs))
        # logger.debug("Length of xs: %s", len(xs))
        # print("Length of ys: ", len(ys))
        # logger.debug("Length of ys: %s", len(ys))

    # Check for None in xs and ys
    # xs = [torch.tensor(x, dtype=torch.long) for x in xs if x is not None]
    # ys = [torch.tensor(y, dtype=torch.long) for y in ys if y is not None]
    xs = [torch.tensor(x, dtype=torch.long) for x in xs if x is not None and len(x) > 0]
    ys = [torch.tensor(y, dtype=torch.long) for y in ys if y is not None and len(y) > 0]
    
    # Check if xs or ys are empty
    if not xs or not ys:
        # print("Warning: xs or ys is empty.")
        logger.warning("xs or ys is empty.")
        return None, None

    # Find max length for zero-padding
    max_len_xs = max([x.size(0) for x in xs])
    max_len_ys = max([y.size(0) for y in ys])

    # Zero-pad
    xs = [F.pad(x, (0, max_len_xs - x.size(0))) for x in xs]
    ys = [F.pad(y, (0, max_len_ys - y.size(0))) for y in ys]
    
    # Convert list of tensors to a single tensor
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    # logger.debug("Final xs: %s", xs)
    # logger.debug("Final ys: %s", ys)

    
    return xs, ys

def split_lines_into_dialogues(lines):
    lines_split = lines.strip().split('\n')
    dialogues = []
    for i in range(0, len(lines_split) - 1, 2):  # Skipping 2 steps to get pairs
        user_line = lines_split[i]
        bot_line = lines_split[i + 1]
        dialogues.append({'user': user_line, 'bot': bot_line})
    return dialogues

def encode_dialogue(dialogue, stoi):
    encoded_dialogue = []
    for key, message in dialogue.items():
        if message is None:
            logger.warning("message for key %s is None.", key)
            continue
        if isinstance(message, list):  # Already encoded
            encoded_message = message
        else:
            encoded_message = [stoi.get(ch, stoi.get('<UNK>')) for ch in message]
        if None in encoded_message:
            logger.warning("Encoded message contains None.")
        encoded_dialogue.extend(encoded_message)
    return encoded_dialogue
# Chatbot part --

# simple tokenization by characters
def encode_0(s, stoi):
    return [stoi[ch] for ch in s]

def encode(s, stoi): # TODO: Check performance
    if isinstance(s, list):  # Check if s is already encoded
        return s
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

    def forward_0(self, x, return_attn_weights=False):
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
    
    def forward(self, x, return_attn_weights=False):
        self.R = self.R.to(x.device)  # Move R to the same device as x
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if self.R.shape[0] < m:  # or x.shape[1], whichever is appropriate
            # Extend self.R dynamically to match x's sequence length
            # extended_R = torch.ones(m, self.R.shape[1]).to(x.device)
            extended_R = torch.ones(m, self.R.shape[1], self.R.shape[2]).to(x.device)
            # extended_R[:self.R.shape[0], :] = self.R
            extended_R[:self.R.shape[0], :, :] = self.R
            self.R = nn.Parameter(extended_R)
            self.R.requires_grad = True  # Enable gradients

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

    def forward_0(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # logger.debug("Shape of x: %s", x.shape)
        # logger.debug("Shape of self.scale: %s", self.scale.shape)
        # logger.debug("Shape of raw: %s", raw.shape)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # logger.debug("Shape of x: %s", x.shape)
        # logger.debug("Shape of self.scale: %s", self.scale.shape)
        # logger.debug("Shape of raw: %s", raw.shape)

        # Extend self.scale dynamically to match x's sequence length
        if self.scale.shape[0] < x.shape[1]:
            extended_scale = torch.ones(x.shape[1], self.scale.shape[1]).to(x.device)
            extended_scale[:self.scale.shape[0], :] = self.scale
            self.scale = nn.Parameter(extended_scale)
            self.scale.requires_grad = True  # Enable gradients

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
            # logger.debug("Shape of logits: %s", logits.shape)
            # logger.debug("Shape of targets: %s", targets.shape)
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

# Train part:
def train_llama_dialogue(device, model, optimizer, dialogues, stoi, config=None, print_logs=False):
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        xs, ys = get_dialogue_batches(dialogues, config['batch_size'], stoi)
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


def train_llama(device, model, optimizer, dataset, scheduler=None, config=None, print_logs=False):
    losses = []
    start_time = time.time()

    # Initialize smallest_loss to a large value
    smallest_loss = float('inf')

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

            # Save model if current_loss is smaller than smallest_loss
            # print('x', x)
            current_loss = x['val']
            if current_loss < smallest_loss:
                old_model_path = f"data/llama_{smallest_loss}.pt"
                
                smallest_loss = current_loss
                
                # Save the new model
                torch.save(model.state_dict(), f"data/llama_{smallest_loss}.pt")

                # Remove old model file if it exists
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            if print_logs:
                # Calculate the remaining time in seconds
                remaining_seconds = batch_time * (config['epochs'] - epoch) / config['log_interval']

                # Create a datetime object for the current time + remaining time
                eta_time = datetime.now() + timedelta(seconds=remaining_seconds)

                # Extract days, hours, minutes and seconds from remaining_seconds
                remaining_time = timedelta(seconds=remaining_seconds)
                days = remaining_time.days
                hours, remainder = divmod(remaining_time.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                # Build a custom remaining time string
                remaining_str_parts = []
                if days > 0:
                    remaining_str_parts.append(f"{days} days")
                if hours > 0:
                    remaining_str_parts.append(f"{hours} hours")
                if minutes > 0:
                    remaining_str_parts.append(f"{minutes} minutes")
                if seconds > 0:
                    remaining_str_parts.append(f"{seconds} seconds")

                remaining_str = " ".join(remaining_str_parts)

                # Format the datetime objects
                eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S")

                # Print the information
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA {eta_str} | Remaining Time {remaining_str}")

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

@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss_dialogue(device, model, dialogues, stoi, config):
    out = {}
    model.eval()
    model.to(device)  # Ensure model is on the correct device
    for split in ["train", "val"]:
        # Assuming you've split your dialogues into train and val
        losses = []
        for _ in range(10):
            xb, yb = get_dialogue_batches(dialogues, config['batch_size'], stoi)
            xb, yb = xb.to(device), yb.to(device)  # Move data to the same device as model
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


@app.get("/")
def read_root():
    return {"message": "Welcome to the Llama API"}

@app.get("/chat/")
def chat(user_input: str):
    global conversation_history
    conversation_history.append(f"<user> {user_input}")
    
    # Generate bot's reply here, using conversation_history as context
    bot_reply = "Hello, I am bot."  # This should be generated by your model
    
    conversation_history.append(f"<bot> {bot_reply}")
    return {"bot_reply": bot_reply}


@app.get("/generate/")
def generate(params: dict = None):
    start_text = params['start_text']
    model_filename = params['model_filename']
    dataset_filepath = params['dataset_filepath']
    MASTER_CONFIG = params['MASTER_CONFIG']
    max_new_tokens = params['max_new_tokens']
    # Record the start time
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device:', device)

    
    # lines = open('./data/input.txt', 'r').read()
    lines = open(f'./{dataset_filepath}', 'r').read()
    vocab = sorted(list(set(lines)))
    itos = {i:ch for i, ch in enumerate(vocab)}
    stoi = {ch:i for i, ch in enumerate(vocab)}
    vocab = sorted(list(set(lines)))

    MASTER_CONFIG['vocab_size'] = len(vocab)

    # Get batches config
    config = {
        'batch_size': 32,
        'context_window': 11,
        'd_model': 13,
    }
    # config = MASTER_CONFIG.copy()

    job_type_train = True

    # dataset = torch.tensor(encode(lines, stoi), dtype=torch.int8)
    dataset = torch.tensor(encode(lines, stoi), dtype=torch.int16)

    llama = Llama(MASTER_CONFIG).to(device)
    optimizer = torch.optim.Adam(llama.parameters())

    print('Loading the model')
    # model_filename = "llama.pt"
    # llama.load_state_dict(torch.load("./data/llama.pt"))
    llama.load_state_dict(torch.load(f"./data/{model_filename}"))


    xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], config)
    xs, ys = xs.to(device), ys.to(device)  # Move data to device

    logits, loss = llama(xs, ys)

    print(loss)

    # Generate part:
    # print(generate(device, llama, MASTER_CONFIG, itos, 500)[0])
    generated_text = generate_from_text(device, llama, MASTER_CONFIG, itos, stoi, start_text=start_text, max_new_tokens=max_new_tokens)
    print(generated_text)

    # Record the end time
    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the time difference
    time_difference = end_time - start_time
    print(f"Time Difference: {str(time_difference)}")

    # generated_text = generate_text(start_text)
    return {"generated_text": generated_text}


@app.post("/train/")
def train(
    MASTER_CONFIG: dict = None
    ):
    dataset_filepath = 'data/telegram_export/input.txt' # TODO: Remove this line
    # Record the start time
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Device:', device)

    lines = open(f'./{dataset_filepath}', 'r').read()
    vocab = sorted(list(set(lines)))
    itos = {i:ch for i, ch in enumerate(vocab)}
    stoi = {ch:i for i, ch in enumerate(vocab)}
    vocab = sorted(list(set(lines)))

    MASTER_CONFIG['vocab_size'] = len(vocab)

    config = {
        'batch_size': 32,
        'context_window': 11,
        'd_model': 13,
    }

    job_type_train = True
    train_type = 'text'  # 'dialogue' or 'text'

    if train_type == 'dialogue':
        raw_dialogues = split_lines_into_dialogues(lines)
        dataset = [{'user': encode(dialogue['user'], stoi), 'bot': encode(dialogue['bot'], stoi)} for dialogue in raw_dialogues]
    else:
        # dataset = torch.tensor(encode(lines, stoi), dtype=torch.int8) # English
        dataset = torch.tensor(encode(lines, stoi), dtype=torch.int16)  # Use int16 or another suitable dtype
        # This is just a sample. The actual code will depend on how you plan to structure the dictionaries.
        # dataset = [{'user': encode(user_line, stoi), 'bot': encode(bot_line, stoi)} for user_line, bot_line in some_function(lines)]

    llama = Llama(MASTER_CONFIG).to(device)
    optimizer = torch.optim.Adam(llama.parameters())

    llama_optimizer = torch.optim.Adam(
        llama.parameters(), 
        betas=(.9, .95), 
        weight_decay=.1, 
        eps=1e-9, 
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

    print('training llama')
    if train_type == 'dialogue':
        train_llama_dialogue(device, llama, optimizer, dataset, stoi, config=MASTER_CONFIG, print_logs=True)
    else:
        train_llama(
            device, 
            llama, 
            optimizer, 
            dataset, 
            scheduler=scheduler,
            config=MASTER_CONFIG, 
            print_logs=True
            )

    print('Saving the model')
    torch.save(llama.state_dict(), "data/llama.pt")

    xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], config)
    xs, ys = xs.to(device), ys.to(device)  # Move data to device

    logits, loss = llama(xs, ys)

    print(loss)

    # Record the end time
    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the time difference
    time_difference = end_time - start_time
    print(f"Time Difference: {str(time_difference)}")

    return {"loss": loss}
