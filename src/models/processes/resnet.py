'''
ResNet process to generate models for the CIFAR-10 dataset.
'''
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Add,
    Activation,
    Input,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout
)

# training parameters
from src.cifar10 import Cifar10
from src.helper import Helper


class Residual:
    @staticmethod
    def block(input_data, filters, conv_size):
        x = Conv2D(filters, conv_size, activation='relu', padding="same")(input_data)
        x = BatchNormalization()(x)
        x = Conv2D(filters, conv_size, activation=None, padding="same")(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_data])
        x = Activation("relu")(x)
        return x


def create_model():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D(3)(x)

    n_resblocks = 10
    for i in range(n_resblocks):
        x = Residual.block(x, 64, 3)

    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    return Model(inputs, outputs)


if __name__ == "__main__":
    helper = Helper()
    cifar10 = Cifar10(dim=3)

    model = create_model()

    # Compilation du modèle
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()

    helper.fit(
        model,
        cifar10.x_train,
        cifar10.y_train,
        1024,
        100,
        validation_data=(cifar10.x_test, cifar10.y_test),
        process_name="resnet"
    )
    # plot_model(m)

    # m.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=1024)
