
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


TEST_NAME = "cifar-10_restnet_epochs=100"

# 47s 940us/sample - loss: 0.7791 - accuracy: 0.7284 - val_loss: 0.7182 - val_accuracy: 0.7595 200 epochs BZ:64 DO:0.2
# 46s 922us/sample - loss: 0.5159 - accuracy: 0.8196 - val_loss: 0.8223 - val_accuracy: 0.7389 100 epochs BZ:64 DO:0.2
# 47s 935us/sample - loss: 0.3512 - accuracy: 0.8780 - val_loss: 0.9876 - val_accuracy: 0.7350 50epoch BZ:64 DO:0.2

# training parameters
BATCH_SIZE = 64
EPOCHS = 80
DROPOUT_RATE = 0.2
FILTERS = 64
NB_CLASS = 10
SKIP = 3


def lr_optimizer(epoch):

    lr = 1e-3
    if epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def Residual_layer(inputs,
                   filters=FILTERS,
                   kernel_size=3,
                   activation=relu,
                   strides=1,
                   batch_norm=True,
                   conf_first=True):

    conv = Conv2D(filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  padding='same')
    x = inputs
    if conf_first:
        x = conv(x)
    else:
        if batch_norm:
            x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
            #x = Dropout(DROPOUT_RATE)(x)
    return x


def resnet_model(input_shape, depth, num_classes=NB_CLASS):

    filter = 16
    num_res_layer = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = Residual_layer(inputs=inputs)
    for stack in range(3):
        for i in range(num_res_layer):
            strides = 1
            if stack > 0 and i ==0:
                strides = 2
            y = Residual_layer(inputs=x,
                               filters=filter,
                               strides=strides)

            y = Residual_layer(inputs=y,
                               filters=filter,
                               activation=None)

            if strides > 0 and i ==0:

                x = Residual_layer(inputs=x,
                                   filters=filter,
                                   kernel_size=1,
                                   strides=strides,
                                   activation=None,
                                   batch_norm=False)
            x = add([x, y])
            x = Activation(relu)(x)
        filter = filter*2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    output = Dense(num_classes,
                   activation=softmax,
                   kernel_initializer='he_normal',
                   name=f"dense_output")(y)

    model = Model(inputs=inputs,outputs=output)

    model.compile(Adam(lr=lr_optimizer(0)),
                  loss=sparse_categorical_crossentropy,
                  metrics=[sparse_categorical_accuracy])

    return  model


if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    input_shape = x_train.shape[1:]
    depth = SKIP * 6 + 2

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    m = resnet_model(input_shape, depth)

    #Compilation du modèle

    print(m.summary())
    plot_model(m, "../../Resnet_model.png")
    tensor_board_callback = TensorBoard(log_dir="logs/"+TEST_NAME)

    history=m.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

    plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
    plt.plot(history.history['sparse_categorical_accuracy'], label='val_accuracy')
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