import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs visible:", tf.config.list_physical_devices("GPU"))