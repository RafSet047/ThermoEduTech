import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.informer.masking import TriangularCausalMask, ProbMask
from src.models.transformer.informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from src.models.transformer.informer.decoder import Decoder, DecoderLayer
from src.models.transformer.informer.attn import FullAttention, ProbAttention, AttentionLayer
from src.models.transformer.informer.embed import DataEmbedding, PositionalEmbedding

from src.shared_state import SharedState
from src.models.model import BaseModel

class Informer(BaseModel):
    def __init__(self, 
                 configs_path: str,
                 state: SharedState,
                 device: str = 'cpu'):
        super().__init__(configs_path, state, device)
        enc_in = state.num_features
        dec_in = state.num_features
        c_out = self._configs.get('c_out', 1)
        seq_len = state.sequence_length
        label_len = seq_len
        out_len = self._configs.get('out_len', 1)
        factor = self._configs.get('factor', 5)
        d_model = self._configs.get('d_model', 128)
        n_heads = self._configs.get('n_heads', 2)
        e_layers = self._configs.get('e_layers', 2)
        d_layers = self._configs.get('d_layers', 2)
        d_ff = self._configs.get('d_ff', 256)
        dropout = self._configs.get('dropout', 0.2)
        attn = self._configs.get('attn', 'prob')
        embed = self._configs.get('embed', 'fixed') 
        freq = self._configs.get('h')
        activation = self._configs.get('gelu')
        output_attention = self._configs.get('output_attention', False)
        distil = self._configs.get('distil', False)
        mix = self._configs.get('mix', False)
        max_len = self._configs.get("max_len", 200)

        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        # Only predict the next value (last time step output)
        dec_out = self.projection(enc_out[:, -1, :])  # (batch_size, c_out)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # (batch_size, c_out)

if __name__ == "__main__":
    pass
    N = 32
    F = 18
    S = 12
    
    x = torch.randn(N, S, F)
    state = SharedState()
    state.num_features = F
    state.sequence_length = S
    model = Informer("", state, 'cpu')
    y = model(x)
    print(y.shape)