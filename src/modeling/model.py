from torch import nn
from src.utils.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES
import numpy as np
import torch
import pandas as pd
from src.modeling.loss import calculate_predictions

def embedding_dim(vocab_size):
    return int(vocab_size**0.25) + 1
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim(vocab_size)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x)
        return self.flatten(x)

class DNNModel(nn.Module):
    def __init__(self, df: pd.core.frame.DataFrame, output_units=3):

        def feat_dict(df):
            features = {k: v.values for k, v in dict(df[CATEGORICAL_FEATURES]).items()}
            features['log_calibration_value'] = df['log_calibration_value'].values
            return features

        feature_dict = feat_dict(df)

        if not isinstance(feature_dict, dict):
            raise TypeError(f'Features must be dictionary. Got {type(feature_dict)} instead.')
        else:
            super(DNNModel, self).__init__()

            self.numeric_features = NUMERIC_FEATURES
            self.categorical_features = CATEGORICAL_FEATURES

            # Embedding layers
            self.embedding_layers = nn.ModuleDict({
                key: EmbeddingLayer(vocab_size=len(np.unique(feature_dict[key])))
                for key in self.categorical_features
            })

            # Calculate total input size
            total_embedding_dim = sum(embedding_dim(len(np.unique(feature_dict[key])))
                                      for key in self.categorical_features)
            total_input_size = len(self.numeric_features) + total_embedding_dim

            # Deep model
            self.deep_model = nn.Sequential(
                nn.Linear(total_input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_units)
            )

    def forward(self, inputs):
        numeric_input = inputs[self.numeric_features[0]]

        embedding_outputs = [self.embedding_layers[key](inputs[key])
                             for key in self.categorical_features]

        deep_input = torch.cat([numeric_input] + embedding_outputs, dim=1)
        return self.deep_model(deep_input)
        
    def predict(self, validation_dataloader):
        self.eval()
        logits = []
        with torch.no_grad():
            for batch, (features, label) in enumerate(validation_dataloader):
                outputs = self.forward(features)
                logits.append(outputs)
        logits = torch.cat(logits)
        return calculate_predictions(logits)
