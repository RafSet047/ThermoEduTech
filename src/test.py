import os
import joblib
from loguru import logger
import tensorflow as tf
from torch import nn
import models
from factory import Factory
from utils import get_instance


def load_model(model_path: str):
    if model_path.endswith('.h5'):
        model = tf.keras.models.load(model_path)
        return model
    elif model_path.endswith('joblib'):
        model = joblib.load(model_path)
    elif model_path.endswith('.pt'):
        torch_models_map = {
            "vanilla": "VanillaTransformer",
            "tab": "TabTransformer",
            "ft": "FTabTransformer",
            "stab": "STabTransformer",
            "informer": "Informer"
        }
        for k, v in torch_models_map:
            if k in model_path:
                model = get_instance(models, torch_models_map[k])
    else:
        logger.error(f"Unsupported model extension: {model_path}")
        raise ValueError(model_path)
    
        