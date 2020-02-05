import matplotlib.pyplot as plt
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
import numpy as np

# tf.compat.v1.keras.layers.CuDNNLSTM

class PrintTrueTrainMetricsAtEpochEnd(Callback):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_train, self.y_train, batch_size=64)
        print(f"The true train loss: {loss}")
        print(f"The true acc loss: {acc}")


def create_model(data_shape):

    m = Sequential()

    m.add(LSTM(128,
               return_sequences=True,
               activation=relu,
               input_shape=(data_shape)))
    m.add(Dropout(0.2))

    m.add(LSTM(128, activation=relu))
    m.add(Dropout(0.2))

    m.add(Dense(32, activation=relu))
    m.add(Dropout(0.1))

    m.add(Dense(10, activation=softmax))

    m.compile(optimizer=Adam(lr=1e-3, decay=1e-5), # reduce learning rate to get best acc
                loss=sparse_categorical_crossentropy,
                metrics=[sparse_categorical_accuracy])

    return m

if __name__ == "__main__":

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    x_train = x_train / 255.0
    y_val = y_val / 255.0

    x_train = np.reshape(x_train, (50000, 32, 96))
    x_val = np.reshape(x_val, (10000, 32, 96))

    data_shape = x_train.shape[1:]
    m = create_model(data_shape=data_shape)

    print(m.summary())
    plot_model(m, "test_lstm.png")

    m.fit(x_train,
          y_train,
          validation_data=(x_val, y_val),
          epochs=50,
          batch_size=64,
          callbacks=[PrintTrueTrainMetricsAtEpochEnd(x_train, y_train)])
