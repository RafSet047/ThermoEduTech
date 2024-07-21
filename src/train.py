import sys
from factory import Factory
from trainer import ModelTrainer
import torch
from torch.utils.data import DataLoader

def train(config_path: str):
    f = Factory(config_path)
    
    train_data = f.create_dataset('train')
    valid_data = f.create_dataset('valid')

    model = f.create_model()

    optimizer = f.create_optimizer(model.parameters())
    criterion = f.create_criterion()
    schedulers = f.create_schedulers(optimizer)

    train_loader = DataLoader(train_data, batch_size=f.get_batch_size(), shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=f.get_batch_size(), shuffle=False)
    
    t = ModelTrainer(model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    schedulers=schedulers
                    )

    train_loss, valid_loss = t.train(f.get_epochs())
    best_weights = t.get_best_weights()
    torch.save(best_weights, f.get_model_path()) 
    
    
if __name__ == "__main__":
    c = sys.argv[1]
    train(c)