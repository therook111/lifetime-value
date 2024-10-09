import torch
from torch import nn
import torch.nn.functional as F

def calculate_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
      logits: [batch_size, 3] tensor of logits.
      
      Each entry of logits contain: [probability, mean, scale]

    Returns:
      preds: [batch_size, 1] tensor of predicted mean.
    """

    logits = logits.to(dtype=torch.float32)

    # Probability that the customer is not churned
    probability = torch.sigmoid(logits[..., :1])

    # Mean value of the customer's LTV
    mean = logits[..., 1:2]

    # Scale parameter of the customer's LTV
    scale = F.softplus(logits[..., 2:])

    # Prediction calculation:
    preds = (
      probability * torch.exp(mean + 0.5*torch.square(scale))
    )
    return preds
class ZILNLoss(nn.Module):

    """
    PyTorch implementation of the Zero-Inflated Lognormal loss function.

    The loss function is designed to handle the scenario where the data is zero-inflated.
    The classification loss component is a binary cross-entropy loss

    The regression loss component measures the error between the predicted mean and the actual mean.

    The overall loss is the sum of the classification loss and the regression loss.

    """
    def __init__(self):
        super(ZILNLoss, self).__init__()

    def forward(self, labels, logits):
        """
        Forward pass of the ZILN loss function.

        Arguments:
          labels: [batch_size, 1] tensor of actual LTV.
          logits: [batch_size, 3] tensor of logits.

        Outputs:
          loss: [batch_size, 1] tensor of loss values.
        """
        labels = labels.to(dtype=torch.float32)
        positive = (labels > 0).float()

        logits = logits.to(dtype=torch.float32) # bz,
        assert logits.shape == labels.shape[:-1] + (3,), "Logits shape is not compatible."

        positive_logits = logits[..., :1]
        classification_loss = F.binary_cross_entropy_with_logits(
            positive_logits, positive, reduction='none').squeeze(-1) # bz,

        mean = logits[..., 1:2]
        scale = F.softplus(logits[..., 2:])

        # Epsilon is the smallest number that can be represented in the data type. This ensures stability.
        scale = torch.maximum(scale, torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))

        stable_labels = positive * labels + (1 - positive) * torch.ones_like(labels)

        LN_distribution = torch.distributions.LogNormal(mean, scale)

        regression_loss = -torch.mean(positive*LN_distribution.log_prob(stable_labels),
                                      axis=-1)

        loss = classification_loss + regression_loss
        return loss # bz,

        
        
        
        






        












        \



        
