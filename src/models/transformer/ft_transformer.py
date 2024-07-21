# THIS CODE IS COPIED FROM
# https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py


import torch
import torch.nn.functional as F
from torch import nn, einsum
from typing import Dict, Optional
from src.models.model import BaseModel
from src.shared_state import SharedState

from einops import rearrange, repeat

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0., device='cpu'):
    return nn.Sequential(
        nn.LayerNorm(dim, device=device),
        nn.Linear(dim, dim * mult * 2, device=device),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim, device=device)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        device = 'cpu'
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim, device=device)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False, device=device)
        self.to_out = nn.Linear(inner_dim, dim, bias = False, device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        device
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, device=device),
                FeedForward(dim, dropout = ff_dropout, device=device),
            ]))
        self.device = device

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns).to(self.device)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types, device):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim, device=device))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim, device=device))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(BaseModel):
    def __init__(
        self,
        configs_path: str,
        state: Optional[SharedState] = None,
        device: str = 'cpu'
    ):
        super().__init__(configs_path, state, device)
        categories = state.num_categories
        num_continuous = state.num_continious
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        dim = self._configs.get("dim", 32)
        depth = self._configs.get("depth", 6)
        heads = self._configs.get("heads", 8)
        dim_head = self._configs.get("dim_head", 16)
        dim_out = self._configs.get("dim_out", 1)
        num_special_tokens = self._configs.get("num_special_tokens", 2)
        attn_dropout = self._configs.get("attn_dropout", 0.)
        ff_dropout = self._configs.get("ff_dropout", 0.)

        # categories related calculations
        self.device = device
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories), device=self.device), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim, device=self.device)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous, device=device)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            device=device
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim, device=device),
            nn.ReLU(),
            nn.Linear(dim, dim_out, device=device)
        )

    def forward(self, x, return_attn = False):
        x_numer, x_categ = x
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens.to(self.device), x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns