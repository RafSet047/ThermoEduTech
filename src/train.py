import os
import argparse
from loguru import logger
from factory import Factory
from trainer import ModelTrainer
import torch
from torch.utils.data import DataLoader

from utils import Visualizer

def train(config_path: str):
    f = Factory(config_path)
    f.create_output_dirpath()
    
    train_data = f.create_dataset('train')
    train_loader = DataLoader(train_data, batch_size=f.get_batch_size(), shuffle=False)
    logger.info("Loaded train data")
    valid_data = f.create_dataset('valid')
    valid_loader = DataLoader(valid_data, batch_size=f.get_batch_size(), shuffle=False)
    logger.info("Loaded valid data")

    model = f.create_model()
    logger.info("Model is created")

    optimizer = f.create_optimizer(model.parameters())
    criterion = f.create_criterion()
    schedulers = f.create_schedulers(optimizer)
    
    t = ModelTrainer(model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    schedulers=schedulers
                    )
    output_dirpath = f.get_output_dirpath()
    print("Starting the training process...")
    train_loss, valid_loss = t.train(f.get_epochs())
    best_weights = t.get_best_weights()
    torch.save(best_weights, os.path.join(output_dirpath, 'model.pt'))
    Visualizer.plot_train_curves(train_loss, valid_loss, os.path.join(output_dirpath, 'train_results.png'))
    logger.info(f"Training has finished in the {output_dirpath}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config-path', help='Path to config path for training', type=str, required=True)
    args = parser.parse_args()
    train(args.config_path)