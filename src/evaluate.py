import os
import json
from typing import *
from copy import deepcopy
import torch
import argparse
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from factory import Factory
from utils import regression_report, write_json, load_json, Visualizer

TEST_PERIOD = 168

def evaluate(configs_path: str, subset: str = 'test', 
             use_inverse_rescale: bool = True, 
             model_output_path: Optional[str] = None,
             prediction_path: Optional[str] = None):
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
    x_copy = deepcopy(X)
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
    if not model_output_path is None:
        if isinstance(x_copy, tuple):
            traced_model = torch.jit.trace(model, (x_copy, ), check_trace=False)
        else:
            traced_model = torch.jit.trace(model, x_copy, check_trace=False)
        traced_model.save(model_output_path)
        logger.info(f"Successfully saved model in the {model_output_path}")

    if not prediction_path is None:
        pd.DataFrame(data={
            "y_true" : np.squeeze(y_true_rescaled, axis=-1),
            "y_pred" : np.squeeze(y_pred_rescaled, axis=-1)
        }).to_csv(prediction_path, index=False)
        logger.info(f"Successfully saved predictions in the {prediction_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config-path', help='Path to config path from training results directory for eval', type=str, required=True)
    parser.add_argument("-s", '--subset', help='Subset of data for evaluation', type=str, required=False, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--inverse-rescale', help='Whether to load the scaler from the training and do the rescaling on data', default=False, action='store_true')
    parser.add_argument("-m", '--model-path', help='Output model path', type=str, required=False, default=None)
    parser.add_argument("-p", '--pred-path', help='Output predictions path', type=str, required=False, default=None)
    args = parser.parse_args()
    evaluate(args.config_path, args.subset, args.inverse_rescale, args.model_path, args.pred_path)