import tensorflow as tf
import numpy as np

from onehot_trainer import OneHotTrainer
from binary_trainer import BinaryTrainer

oh_trainer = OneHotTrainer()
bin_trainer = BinaryTrainer()

print("onehot train")
oh_trainer.train()

print("binary train")
bin_trainer.train()