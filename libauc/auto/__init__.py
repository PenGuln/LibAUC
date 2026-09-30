"""
AutoMAX: automated hyperparameter optimisation for LibAUC losses/optimizers,
built on SMAC3.

Requires the optional dependencies: ``pip install libauc[auto]``.
"""
from .configuration import AutoMAXConfigration, AutoMAXConfiguration
from .automax import AutoMAX
from .autopartial import autopartial
from .helpers import parse_hyperparameters_from_dict
from .custom_datasets import load_custom_csv_dataset
from .session import SessionAwareAutoMAX, TrainingCancelled
