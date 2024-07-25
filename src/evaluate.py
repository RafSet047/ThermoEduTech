import os
import json
import torch
import argparse
import numpy as np
from loguru import logger
from factory import Factory
from utils import regression_report, write_json

def evaluate(configs_path: str, subset: str = 'train'):
    f = Factory(configs_path)
    
    dataset = f.create_dataset(subset)
    model = f.create_model()
    
    model.load_state_dict(torch.load(f.get_model_path()))
    model.eval()

    X, y_true = dataset.get_dataset()
    y_pred = None
    with torch.no_grad():
        y_pred = model(X)

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().detach().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().detach().numpy()

    results = regression_report(y_true, y_pred)
    
    logger.info(f"Results of the : {f.get_model_path()} model in `test` data") 
    logger.info(json.dumps(results, indent=2))

    write_json(os.path.join(f.get_output_dirpath(), 'test_results.json'), results)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config-path', help='Path to config path from training results directory for eval', type=str, required=True)
    parser.add_argument("-s", '--subset', help='Subset of data for evaluation', type=str, required=False, default='test', choices=['train', 'valid', 'test'])
    args = parser.parse_args()
    evaluate(args.config_path, args.subset)