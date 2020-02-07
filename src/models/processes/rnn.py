'''
RNN process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


def create_model(n_layers, optimizer, n_neurons, dropout):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if dropout is not None:
                model.add(LSTM(n_neurons, dropout=dropout, return_sequences=True, input_dim=(32 * 32 * 3)))
            else:
                model.add(LSTM(n_neurons, return_sequences=True, input_dim=(32 * 32 * 3)))
        else:
            if dropout is not None:
                model.add(LSTM(n_neurons, dropout=dropout, return_sequences=True))
            else:
                model.add(LSTM(n_neurons, return_sequences=True))
    model.add(Dense(10, activation="sigmoid"))
    model.compile(
        optimizer=optimizer,
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model
