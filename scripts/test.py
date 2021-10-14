import os
print(f"Cuda visible devices at the start of the python script: {os.environ['CUDA_VISIBLE_DEVICES']}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
