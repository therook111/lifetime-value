import torch
from src.utils.early_stopping import EarlyStopping
from src.utils.constants import BATCH_SIZE
from tqdm import tqdm 
import warnings

def _train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (features, label) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(features)
        loss = loss_fn(label.unsqueeze(1), outputs)
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if batch == 49:
            print('\n')
        if batch % 50 == 49:
            loss, current = loss.item(), batch * BATCH_SIZE + len(features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end='\n')

def _validation_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (features, label) in enumerate(dataloader):
            outputs = model(features)
            test_loss += loss_fn(label.unsqueeze(1), outputs).mean().item()
    
    test_loss /= num_batches
    print("Validation set: Loss: {:.5f}".format(test_loss))
    
    return test_loss
            

class ModelTrainer:
    def __init__(self, model=None, loss_fn=None, optimizer=None, 
                  train_dataloader=None, test_dataloader=None, 
                  epochs=10, early_stopping: EarlyStopping = None):

        '''Arbitrary class of a model trainer.
        Args:
            - model: A PyTorch model.
            - loss_fn: Loss function
            - optimizer: Optimizer
            - train_dataloader: DataLoader for the training dataset.
            - test_dataloader: DataLoader for the validation_dataset.
            - epochs: Number of epochs to train for
            - early_stopping: An object of class EarlyStopping to implement an early stopping mechanism.
        '''
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer 
        self.early_stopping = early_stopping 
        self.epochs = epochs 
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        if not self.model:
            raise Exception("Model not specified.")
        if not self.loss_fn:
            raise Exception("Loss function not specified.")
        if not self.optimizer:
            raise Exception("Optimizer not specified.")
        if not self.train_dataloader:
            raise Exception("Training DataLoader not specified.")
        if not self.test_dataloader:
            raise Exception("Testing DataLoader not specified")
        
        if self.early_stopping is None or self.early_stopping.patience > self.epochs:
            warnings.warn("""EarlyStopping mechanism's patience is set to be higher than number of training epochs. 
            The mechanism will not do anything.""", UserWarning)

        for i in tqdm(range(self.epochs)):
            _train_model(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
            validation_loss = _validation_loop(self.test_dataloader, self.model, self.loss_fn)

            if self.early_stopping is not None or self.early_stopping.patience < self.epochs:
                self.early_stopping(validation_loss, self.model)
                if self.early_stopping.early_stop:
                    break

        print("Done!")