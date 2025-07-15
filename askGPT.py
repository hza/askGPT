import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# FFN activation function
class Activation(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# Feed-Forward Network (FFN) - Two Linear Layers with GELU
class FFN(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.activation = Activation()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Masked Multi-Head Attention (MHA)
class MHA(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head

        # Linear projections for q, k, v
        self.wq = nn.Linear(d_model, d_model)  # query
        self.wk = nn.Linear(d_model, d_model)  # key
        self.wv = nn.Linear(d_model, d_model)  # value

        # self.qkv = nn.Linear(d_model, 3 * d_model)

        self.wo = nn.Linear(d_model, d_model)  # output
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()  # batch size, sequence length, embedding dim

        # Compute q, k, v (batch_size, seq_len, d_model)
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)

        # alternative optimization: q=x*wq, k=x*wk, v=x*wq
        # qkv = self.qkv(x)
        # q, k, v = qkv.split(d_model, dim=2)

        # Reshape to split heads: 
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim) 
        # d_model = n_head * d_model // n_head
        # Then transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (d_model ** 0.5)) # Q·Kᵀ / sqrt(d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax + weighted sum (batch_size, num_heads, seq_len, head_dim)
        attention_weights = F.softmax(attn, dim=-1)

        x = attention_weights @ v
        
        # Merge heads: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection
        x = self.wo(x)
        x = self.dropout(x)
        return x


# Transformer Block: MHA + FFN + LayerNorm
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_head, dropout_rate)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x

# Sinusoidal positional torch
def sinusoidal_embedding(max_pos, d_model):
    pe = torch.zeros(max_pos, d_model)
    position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: (max_pos, d_model)

# GPT Model
class GPT(nn.Module):
    def __init__(self, wte, d_model, block_size, n_layer, n_head, dropout_rate=0.1):
        super().__init__()
        max_pos = 1024

        self.block_size = block_size
        self.dropout = nn.Dropout(dropout_rate)
        # Use sinusoidal positional embeddings instead of learned ones
        wpe = nn.Embedding(max_pos, d_model)
        wpe.weight.data = sinusoidal_embedding(max_pos, d_model)
        wpe.weight.requires_grad = False

        self.wpe = wpe     # positional embeddings
        self.wte = wte     # token embeddings

        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_head, dropout_rate) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, wte.num_embeddings, bias=False)

    def forward(self, idx):
        batch_size, d_model = idx.size()

        # creates a 1D tensor (vector) of integers from 0 to d_model-1
        # unsqueeze is a function that adds a dummy dimension (of size 1)
        pos = torch.arange(d_model, dtype=torch.long, device=idx.device).unsqueeze(0)

        # Token + Positional Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Transformer blocks
        x = self.blocks(x)

        # Final layer norm and output logits
        x = self.ln_f(x)
        x = self.dropout(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1, top_k=None):
        # Take a conditioning sequence of indices (LongTensor of shape (b,t)) and complete
        # the sequence max_new_tokens times, feeding predictions back into input.
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            # if the sequence context is growing too long, crop it
            tokens_cond = idx if idx.size(1) <= self.wpe.num_embeddings else idx[:, -self.wpe.num_embeddings:]

            # get the predictions
            logits = self(tokens_cond)
            
            # pluck the logits at the final step and scale by temperature
            # last logit in the sequence
            # [batch_size, sequence_length, vocab_size] -> [batch_size, vocab_size]
            logits = logits[:, -1, :] / temperature

            # optionally crop probabilities to only top-k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                # PyTorch broadcasting feature
                # v[:, [-1]] creates a 2D tensor with the last value of each row
                # logits < v[:, [-1]] creates a mask where logits are less than the last value in each row
                # logits[mask] = -float('Inf') sets those logits to negative infinity
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1) # by vocab_size

            # sample from the distribution (num_samples=1 means one index)
            idx_next = torch.multinomial(probs, num_samples=1)

            # just the most expected value
            # idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # Training function for the GPT model
    # This function takes a tokenizer, a list of texts, and training parameters
    # It tokenizes the texts, creates a DataLoader, and trains the model using AdamW    
    def train(self, tokenizer, texts, device, epochs=20, batch_size=1, lr=5e-4):
        dataset = TextDataset(tokenizer, texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Adam Optimizer (Adaptive Moment Estimation)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                input_ids, target_ids = [b.to(device) for b in batch]

                logits = self.forward(input_ids)

                # logits.view(-1, logits.size(-1)):
                #   [batch_size, sequence_length, vocab_size] → [(batch_size × sequence_length), vocab_size]
                # This is the standard way to compute the loss for language models
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),   # → [batch_size × seq_length, vocab_size]
                    target_ids.view(-1))                # → [batch_size × seq_length]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")


class WordTokenizer:
    def __init__(self, texts):
        text = ' '.join(texts)
        words = sorted(set(text.split()))
        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi.get(w, self.vocab_size) for w in text.split()]  # unknown words get id = vocab_size

    def decode(self, token_ids):
        return ' '.join([self.itos.get(i.item(), "[UNK]") for i in token_ids])

class WordEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, d_model)  # +1 for [UNK]

    def forward(self, token_ids):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return self.embeddings(token_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenized_texts = [torch.tensor(tokenizer.encode(text), dtype=torch.long) 
                             for text in texts]

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        chunk = self.tokenized_texts[idx]
        # [all tokens except the last one]  [all tokens except the first one]
        return chunk[:-1], chunk[1:]  # input, target
    

texts = [
    "London is the capital of England <eos> ", 
    "Paris is the capital of France yeah <eos> ", 
    "Berlin is the capital of Germany <eos> ",
    "Madrid is the capital of Spain <eos> ",
    "Best city of France ? Paris is the best city <eos> ", 
    "What is Germany ? Germany is a country <eos> ", 
    "What is England ? England is a country <eos> ", 
    "What is France ? France is a country <eos> ", 
    "What is the capital of England ? London is the capital of England <eos> ", 
    "What is the capital of Germany ? Berlin is the capital of Germany <eos> ", 
    "What is the capital of Spain ? Madrid is the capital of Spain <eos> ", 
    "What is Paris ? Paris is a city in France <eos> ",
]

d_model=256

tokenizer = WordTokenizer(texts)
embedder = WordEmbedder(vocab_size=tokenizer.vocab_size, d_model=d_model)

# GPT model with token embeddings, positional embeddings, and transformer blocks
# d_model is the dimension of the model (embedding size)
# block_size is the maximum sequence length the model can handle
# n_layer is the number of transformer blocks, n_head is the number of attention heads
# dropout_rate is the dropout rate applied in the model
model = GPT(wte=embedder.embeddings, d_model=d_model, block_size=64, n_layer=8, n_head=8, dropout_rate=0.1)

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Encode prompt
prompt = "What is the capital of France ?"
prompt_ids = tokenizer.encode(prompt)

# Train the model
# Note: This is a simplified training loop for demonstration purposes.
# In practice, you would want to use a more robust training loop with validation, checkpointing, etc.
model.train(tokenizer, texts, device, epochs=25)

input_ids = torch.tensor(prompt_ids, dtype=torch.long).reshape(1, len(prompt_ids))

for i in range(10):
    print(f"Try # {i}")
    output_ids = model.generate(input_ids, max_new_tokens=15, temperature=1, top_k=None)
    print(tokenizer.decode(output_ids[0]))
