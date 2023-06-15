import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from audioProcessor import AudioProcessor, Audio
import matplotlib.pyplot as plt
import seaborn as sns


from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing, BatchNormalization
from keras.optimizers import Adam,RMSprop
from keras.losses import SparseCategoricalCrossentropy


def load_data(spectogram_type):
    train_X = np.array([np.load(f"data/spectograms/{spectogram_type}/train/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    train_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    cv_X = np.array([np.load(f"data/spectograms/{spectogram_type}/cv/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    cv_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    test_X = np.array([np.load(f"data/spectograms/{spectogram_type}/test/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])
    test_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])

    return train_X, train_Y, cv_X, cv_Y, test_X, test_Y


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


def train_model(model, train_X, train_Y, cv_X, cv_Y):
    return model.fit(train_X, train_Y, epochs=10, validation_data=(cv_X, cv_Y))



def plot_accuracy_loss(history):
    metrics = history.history
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(history.epoch, metrics['loss'], metrics['val_loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].legend(['train loss', 'validation loss'])


    ax[1].plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylim([0, 1])
    ax[1].legend(['train accuracy', 'validation accuracy'])

    fig.set_size_inches(8, 5)
    plt.show()


def plot_confusion_matrix(true_categories, predicted_categories):
    cm = confusion_matrix(true_categories, predicted_categories)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d',cmap='crest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def run_model(model, train_X, train_Y, cv_X, cv_Y, test_X, test_Y):
    history = train_model(model, train_X, train_Y, cv_X, cv_Y)

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)

    print(f"Test accuracy: {test_acc}")

    model.summary()

    plot_accuracy_loss(history)

    # Predict the test data
    y_pred = model.predict(test_X)
    predicted_categories = np.argmax(y_pred, axis=1)
    true_categories = test_Y

    plot_confusion_matrix(true_categories, predicted_categories)

def save_model(model, model_name):
    model.save(f"models/{model_name}.h5")




def run(spectogram_type="mfcc"):

    train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data(spectogram_type)

    run_model(model_4(train_X), train_X, train_Y, cv_X, cv_Y, test_X, test_Y)


    
        
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Train a model")
    argParser.add_argument("--spectogram_type", type=str, help="Type of spectogram to use", required=True,  choices=["mfcc","melspectrogram","chroma_stft"])
    args = argParser.parse_args()
    run(args.spectogram_type)
    