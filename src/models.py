from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing, BatchNormalization
from keras.optimizers import Adam,RMSprop
from keras.losses import SparseCategoricalCrossentropy


def model_1(train_X):

    model = Sequential(
        [
            Input(shape=(*train_X[0].shape,1)),
            Resizing(32, 32), 
            Conv2D(32, 3, strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Conv2D(128, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax'),
        ]
    )
    model.compile(optimizer=RMSprop(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model


def model_2(train_X):

    model = Sequential(
        [
            Input(shape=(*train_X[0].shape,1)),
            Resizing(32, 32), 
            Conv2D(32, 3, strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Conv2D(128, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax'),
        ]
    )
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model

def model_3(train_X):
    model = Sequential(
    [
        Input(shape=(*train_X[0].shape,1)),
        Resizing(32, 32), 
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        #Conv2D(64, 3, activation='relu'),
        #MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10),
    ]
    )
    model.compile(optimizer=RMSprop(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model


def model_4(train_X):
    model = Sequential(
    [
        Input(shape=(*train_X[0].shape,1)),
        Resizing(32, 32), 
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        #Conv2D(64, 3, activation='relu'),
        #MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10),
    ]
    )
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model


def model_5(train_X):
    model = Sequential(
    [
        Input(shape=(*train_X[0].shape,1)),
        Resizing(32, 32), 
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        # Conv2D(128, 3, activation='relu'),
        # MaxPooling2D(),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10),
    ]
    )
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model