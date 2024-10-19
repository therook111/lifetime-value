from src.utils.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader 
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import numpy as np 


def split_train_val(df: pd.core.frame.DataFrame, test_size=0.2, random_state=42):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError(f'df must be Pandas DataFrame. Got {type(df)} instead.')
    else:
        for key in CATEGORICAL_FEATURES:
            encoder = LabelEncoder()
            df[key] = encoder.fit_transform(df[key])
        
        y0 = df['calibration_value'].values
        df_train, df_eval, y0_train, y0_eval = train_test_split(
        df, y0, test_size=test_size, random_state=random_state)

        def feature_dict(df):
            features = {k: v.values for k, v in dict(df[CATEGORICAL_FEATURES]).items()}
            features['log_calibration_value'] = df['log_calibration_value'].values
            return features

        X_train, y_train = feature_dict(df_train), df_train['label'].values
        X_eval, y_eval = feature_dict(df_eval), df_eval['label'].values

        return X_train, X_eval, y_train, y_eval, y0_eval

class CustomerDataset(Dataset):
    def __init__(self, features: dict, label=None):
        if not isinstance(features, dict):
            raise TypeError(f'Features must be dictionary. Got {type(features)} instead.')
        else:
            self.features = features
            self.label = label

    def __len__(self):
        if self.label is not None:
            return len(self.label)
        else:
            return len(next(iter(self.features.values())))
            
    def __getitem__(self, idx):

        sample = {key: torch.tensor(values[idx], dtype=torch.int).unsqueeze(0) 
                  for key, values in self.features.items()
                  if key not in NUMERIC_FEATURES}
        
        for feature in NUMERIC_FEATURES:
            sample[feature] = torch.tensor(self.features[feature][idx], dtype=torch.float32).unsqueeze(0)
        
        if self.label is not None:
            label = self.label[idx]
            return sample, label
        else:
            return sample 

def _to_torch_dataset(features, label, batch_size=1024, train=True):

    '''
    Utility function to convert the data into PyTorch's compatible tensors

    Arguments:
        - X: Features of the data. Accept a dictionary of features in the form: {'features': values}, a numpy ndarray, a pandas DataFrame or a Python list
        - y: Label of the data. Accepts a numpy ndarray or a Python list.
        - batch_size: Batch size for the DataLoader. Default: 32
        - train: If set to True, the dataset is for training. If set to False, the dataset is for evaluation.
        The parameter is necessary to decide whether to shuffle at every batch for the DataLoader
    '''
    
    if not isinstance(features, (dict, pd.core.frame.DataFrame, np.ndarray, list)):
        raise TypeError(f'Features must be dictionary, NumPy ndarray, Python list, or Pandas DataFrame. Got {type(features)} instead.')
    if not isinstance(label, (np.ndarray, list)):
        raise TypeError(f'Label must be NumPy array or Python list. Got {type(label)} instead.')
    if not isinstance(batch_size, int):
        raise TypeError(f'Batch size must be an integer. Got {type(batch_size)} instead')

    dataset = CustomerDataset(features, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataset, dataloader

    


        
        

        

        
