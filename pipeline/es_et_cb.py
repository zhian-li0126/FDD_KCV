from typing import Any
import torch
import joblib

class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False
class Callback:
    def __init__(self):
        self.best_val_acc = 0
        self.best_model_state = None

    def check_and_save_best_model(self, model, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            if isinstance(model, torch.nn.Module):
                self.best_model_state = model.state_dict()    
            else:
                self.best_model_state = model            

    def save_best_model(self, path):
        if self.best_model_state:
            try:
                torch.save(self.best_model_state, path)
            except:
                joblib.dump(self.best_model_state, path)
        else:
            print("No model state to save.")
    def __call__(self) -> None:
        print("Eureka")
        print(torch.rand(1))