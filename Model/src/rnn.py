import matplotlib.pyplot as plt
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import plot_model
from tensorflow_core.python.keras.layers import CuDNNLSTM
import numpy as np

TEST_NAME = "tf2_RNN_epochs=200"
#13s 259us/sample - loss: 0.7278 - sparse_categorical_accuracy: 0.7569 - val_loss: 1.4812 - val_sparse_categorical_accuracy: 0.6175
# 14s 271us/sample - loss: 0.7001 - accuracy: 0.7708 - val_loss: 1.4332 - val_accuracy: 0.6244 epochs=100, batch_size=32
# tf.compat.v1.keras.layers.CuDNNLSTM


class PrintTrueTrainMetricsAtEpochEnd(Callback):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_train, self.y_train, batch_size=64)
        print(f"Le Vrai loss du train : {loss}")
        print(f"La Vrai acc du train : {acc}")


def create_model(data_shape):

    m = Sequential()

    m.add(CuDNNLSTM(128,
               return_sequences=True,
                # activation=relu
               input_shape=(data_shape)))
    m.add(Dropout(0.5))

    m.add(CuDNNLSTM(128)) # activation=relu)) # if use CuDNNLSTM don't use activation
    m.add(Dropout(0.5))

    m.add(Dense(64, activation=relu, kernel_regularizer=L1L2(l2=0.1)))
    #m.add(Dropout(0.5))

    m.add(Dense(32, activation=relu, kernel_regularizer=L1L2(l2=0.1)))
    #m.add(Dropout(0.4))

    m.add(Dense(10, activation=softmax))

    m.compile(optimizer=Adam(), # reduce learning rate to get best acc if epochsMax/2 ==> lr <<
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

    return m

if __name__ == "__main__":

    tensor_board_callback = TensorBoard("./logs/" + TEST_NAME)

    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    # reshape par ligne
    x_train = np.reshape(x_train, (50000, 32, 96)) / 255.0
    x_val = np.reshape(x_val, (10000, 32, 96)) /255.0

    data_shape = x_train.shape[1:]

    m = create_model(data_shape=data_shape)

    print(m.summary())
    plot_model(m, "test_lstm.png")

    history = m.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=200,
                    batch_size=64,
                    callbacks=[PrintTrueTrainMetricsAtEpochEnd(x_train, y_train)])

    plt.plot(history.history['categorical_accuracy'], label='categorical_accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='val_categorical_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = m.evaluate(x_val, y_val, verbose=2)
    print(test_acc)
    # Testing predictions
    predict = m.predict(x_val)
    print(predict[0])
    print(np.argmax(predict[0]))
