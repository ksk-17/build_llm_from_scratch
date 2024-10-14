# custom function for handling imports

# required imports
import torch
import torch.nn as nn


# required classes
class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qvk_bias = False):
    super().__init__()
    assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.W_key = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.W_value = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)

    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    attn_scores = torch.matmul(queries, keys.transpose(2, 3))
    masked_attn_scores = attn_scores.masked_fill_(self.mask == 0, -torch.inf)
    attn_weights = nn.functional.softmax(masked_attn_scores/(keys.shape[-1] ** 0.5), dim = -1)
    attn_weights = self.dropout(attn_weights)
    context_vector = torch.matmul(attn_weights, values).transpose(1, 2)
    context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
    context_vector = self.out_proj(context_vector)
    return context_vector
  
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1, keepdim = True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift
  
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
  
class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
        GELU(),
        nn.Linear(4 * config["emb_dim"], config["emb_dim"])
    )

  def forward(self, x):
    return self.layers(x)
  
class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = MultiHeadAttention(
        d_in = config["emb_dim"],
        d_out = config["emb_dim"],
        context_length = config["context_length"],
        num_heads = config["n_heads"],
        dropout = config["drop_rate"],
        qvk_bias = config["qvk_bias"]
    )
    self.ffn = FeedForward(config)
    self.norm1 = LayerNorm(config["emb_dim"])
    self.norm2 = LayerNorm(config["emb_dim"])
    self.dropout_skip = nn.Dropout(config["drop_rate"])

  def forward(self, x):
    skip = x
    x = self.norm1(x)
    x = self.attn(x)
    x = self.dropout_skip(x)
    x = x + skip

    skip = x
    x = self.norm2(x)
    x = self.ffn(x)
    x = self.dropout_skip(x)
    x = x + skip
    return x
  
class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
    self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
    self.dropout = nn.Dropout(config["drop_rate"])
    self.transformer_blocks = nn.Sequential(
        *[TransformerBlock(config) for _ in range(config["n_layers"])]
    )
    self.final_norm = LayerNorm(config["emb_dim"])
    self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len))
    x = tok_embeds + pos_embeds
    x = self.dropout(x)
    x = self.transformer_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
  

# required helper classes
def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)
    logits = logits[:, -1, :]
    probs = nn.functional.softmax(logits, dim = -1)
    idx_next = torch.argmax(probs, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim = -1)
  return idx