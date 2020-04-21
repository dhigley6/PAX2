
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def tf_train_step(deconvolved, optimizer):
    with tf.GradientTape() as tape:
        reconvolved = tf.signal.convolve()
        loss = mean_squared_error(measurement, reconvolved)
    grads = tape.gradient(loss, deconvolved)
    optimizer.apply_gradients(zip(deconvolved, last))