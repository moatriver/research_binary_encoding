from sre_constants import BIGCHARSET
import tensorflow as tf
import numpy as np

from onehot_trainer import OneHotTrainer
from binary_trainer import BinaryTrainer

# def my_cnn(inputs, output_dim):
#     # imitate resnet
#     x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
# 
#     x = tf.keras.layers.Dense(128)(x)
#     x = tf.keras.layers.LeakyReLU()(x)
#     x = tf.keras.layers.Dense(output_dim)(x)
# 
#     return x

oh_trainer = OneHotTrainer()
bin_trainer = BinaryTrainer()

print("onehot train")
oh_trainer.train()

print("binary train")
bin_trainer.train()