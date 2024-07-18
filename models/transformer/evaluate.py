import os
import sys

from data_loader import TransformerTabData
from ft_transformer import FTTransformer

import torch
from torch.utils.data import DataLoader

def eval(data_path: str, data_configs_path: str, model_path: str):
    dataset = TransformerTabData(data_path, data_configs_path)
    data = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    (X_num, X_cat), Y = next(iter(data))
    print(X_num.shape)
    print(X_cat.shape)
    print(Y.shape)

    model = FTTransformer(categories=dataset.num_categories, 
                          num_continuous=dataset.num_continious,
                          heads=8, dim=32, dim_out=1, depth=6,
                          attn_dropout=0.2, ff_dropout=0.2,
                          device='cpu'
                          )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred = model(X_cat, X_num)
    print(pred)

if __name__ == "__main__":
    d_path = sys.argv[1]
    c_path = sys.argv[2]
    m_path = sys.argv[3]
    eval(d_path, c_path, m_path)