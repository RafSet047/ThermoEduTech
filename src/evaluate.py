import os
import json
import torch
import argparse
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from factory import Factory
from utils import regression_report, write_json, load_json, Visualizer

TEST_PERIOD = 168

def evaluate(configs_path: str, subset: str = 'train', use_inverse_rescale: bool = True):
    f = Factory(configs_path)
    
    dataset = f.create_dataset(subset)
    dataset.prepare_data()

    scaler = None
    if use_inverse_rescale:
        scaler_data_path = dataset.get_scaler_data_path()
        scaler_data = load_json(scaler_data_path)
        scaler = joblib.load(scaler_data['y_scaler_path'])
    
    model = f.create_model()
    model.load_state_dict(torch.load(f.get_model_path()))
    model.eval()

    X, y_true = dataset.get_dataset()
    y_pred = None
    with torch.no_grad():
        y_pred = model(X)

    y_pred_rescaled, y_true_rescaled = None, None
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().detach().numpy()
        if not scaler is None:
            y_true_rescaled = scaler.inverse_transform(y_true.reshape(-1, 1))
        else:
            y_true_rescaled = y_true
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().detach().numpy()
        if not scaler is None:
            y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
        else:
            y_pred_rescaled = y_pred

    results = regression_report(y_true, y_pred)
    
    Visualizer.plot_predictions(y_true=y_true_rescaled[-TEST_PERIOD:, 0], 
                                y_pred=y_pred_rescaled[-TEST_PERIOD:, 0], 
                                save_path=os.path.join(f.get_output_dirpath(), 'actual_vs_pred.png')) 
    logger.info(f"Results of the : {f.get_model_path()} model in `test` data") 
    logger.info(json.dumps(results, indent=2))

    write_json(os.path.join(f.get_output_dirpath(), 'test_results.json'), results)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config-path', help='Path to config path from training results directory for eval', type=str, required=True)
    parser.add_argument("-s", '--subset', help='Subset of data for evaluation', type=str, required=False, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--inverse-rescale', help='Whether to load the scaler from the training and do the rescaling on data', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args.config_path, args.subset, args.inverse_rescale)