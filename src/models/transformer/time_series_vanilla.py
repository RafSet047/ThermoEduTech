# This is code idea is taken from
# https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py

import torch
import torch.nn as nn
import math
from typing import Optional
from src.models.model import BaseModel
from src.shared_state import SharedState

from loguru import logger

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class VanillaTransformer(BaseModel):
    #def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, max_len, dropout=0.1):
    def __init__(self, 
                 configs_path: str,
                 state: Optional[SharedState] = None,
                 device: str = 'cpu',
                ):
        super().__init__(configs_path, state, device)

        num_features = state.num_features
        d_model = self._configs.get('d_model', 32)
        nhead = self._configs.get('nhead', 4)
        num_encoder_layers = self._configs.get('num_encoder_layers', 2)
        dim_feedforward = self._configs.get('d_ffn', 64)
        max_len = self._configs.get('max_len', 200)
        dropout = self._configs.get('dropout', 0.2)
        d_output = self._configs.get('d_output', 1)

        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, d_output)
        self.d_model = d_model
        self.init_weights()

        logger.info("VanillaTransformer is initialized")

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, src_padding_mask=None):
        # src shape: (batch_size, seq_len, num_features)
        src = self.embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)  # (batch_size, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src, src_mask, src_padding_mask)  # (seq_len, batch_size, d_model)
        output = output.transpose(0, 1)  # (batch_size, seq_len, d_model)
        output = self.decoder(output[:, -1, :])  # (batch_size, 1)
        return output#.squeeze(-1)  # (batch_size,)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

if __name__ == "__main__":
    N = 32
    F = 18
    S = 12
    
    x = torch.randn(N, S, F)
    
    state = SharedState()
    state.num_features = F
    model = VanillaTransformer(
        "",
        state,
        'cpu'
    )

    y = model(x)
    print(y.shape)