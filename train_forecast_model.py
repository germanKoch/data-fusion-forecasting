import pandas as pd
import lightning.pytorch as pl
import torch

from tqdm import tqdm
from matplotlib import pyplot as plt
from metrics.metrics import RMSLE
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

def load_data():
    target_series = pd.read_parquet('data/target_series.parquet')
    calendar = pd.read_csv('data/calendar.csv', parse_dates=['date'])
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['date'] = calendar['date'].dt.tz_localize(None)

    train_calendar = calendar[calendar['part'] == 'train'][['date', 'week']]
    private_calendar = calendar[calendar['part'] == 'private'][['date', 'week']]
    public_calendar = calendar[calendar['part'] == 'public'][['date', 'week']]
    return target_series, train_calendar
    
    
torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if hasattr(torch, 'mps'):
    torch.mps.manual_seed(1337)

hyperparams = dict(
    min_encoder_length=12,
    max_encoder_length=2*12,
    min_prediction_length=1,
    max_prediction_length=12,
    batch_size=1024,
    quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    learning_rate=3e-5,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,    
)



