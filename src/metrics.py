import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Sequence
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def _cumulative_true(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> np.ndarray:
  """Calculates cumulative sum of lifetime values over predicted rank.

  Arguments:
    y_true: true lifetime values.
    y_pred: predicted lifetime values.

  Returns:
    res: cumulative sum of lifetime values over predicted rank.
  """
  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
  }).sort_values(
      by='y_pred', ascending=False)

  return (df['y_true'].cumsum() / df['y_true'].sum()).values

def get_gain_charts(y_true, y_pred, y0_true):
    gain = pd.DataFrame({
    'lorenz': _cumulative_true(y_true, y_true),
    'baseline': _cumulative_true(y_true, y0_true),
    'model': _cumulative_true(y_true, y_pred),
})
    return gain


def gini_from_gain(y_true, y_pred, y0_true) -> pd.DataFrame:
  """Calculates gini coefficient over gain charts.

  Arguments:
    df: Each column contains one gain chart. First column must be ground truth.
    y_true: True LTV
    y_pred: Prediction of model
    y0_true: First-time purchase value.

  Returns:
    gini_result: This dataframe has two columns containing raw and normalized
                 gini coefficient.
  """

  df = get_gain_charts(y_true, y_pred, y0_true)

  raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
  normalized = raw / raw[0]

  return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
  })[['raw', 'normalized']]


def _normalized_rmse(y_true, y_pred):
  return np.sqrt(metrics.mean_squared_error(y_true, y_pred)) / y_true.mean()


def _normalized_mae(y_true, y_pred):
  return metrics.mean_absolute_error(y_true, y_pred) / y_true.mean()


def _aggregate_fn(df):
  return pd.Series({
      'label_mean': np.mean(df['y_true']),
      'pred_mean': np.mean(df['y_pred']),
      'normalized_rmse': _normalized_rmse(df['y_true'], df['y_pred']),
      'normalized_mae': _normalized_mae(df['y_true'], df['y_pred']),
  })


def decile_stats(
    y_true: Sequence[float],
    y_pred: Sequence[float]) -> pd.DataFrame:
  """Calculates decile level means and errors.

  The function first partites the examples into ten equal sized
  buckets based on sorted `y_pred`, and computes aggregated metrics in each
  bucket.

  Arguments:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    df: Bucket level statistics.
  """
  num_buckets = 10
  decile = pd.qcut(
      y_pred, q=num_buckets, labels=['%d' % i for i in range(num_buckets)])

  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
      'decile': decile,
  }).groupby('decile').apply(_aggregate_fn)

  df['decile_mape'] = np.abs(df['pred_mean'] -
                             df['label_mean']) / df['label_mean']
  return df

def spearmanr(x1: Sequence[float], x2: Sequence[float]) -> float:
  """Calculates spearmanr rank correlation coefficient.

  See https://docs.scipy.org/doc/scipy/reference/stats.html.

  Args:
    x1: 1D array_like.
    x2: 1D array_like.

  Returns:
    correlation: float.
  """
  return stats.spearmanr(x1, x2, nan_policy='raise')[0]


def get_metric_report(y_true, y_pred, y0_true, company=None):
    gini = gini_from_gain(y_true, y_pred, y0_true)
    df_decile = decile_stats(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)

    report = pd.DataFrame(
    {
        'company': company,
        'label_mean': y_true.mean(),
        'pred_mean': y_pred.mean(),
        'label_positive': np.mean(y_true > 0),
        'decile_mape': df_decile['decile_mape'].mean(),
        'baseline_gini': gini['normalized'][1],
        'gini': gini['normalized'][2],
        'spearman_corr': spearman,
    },
    index=[0])

    return report

