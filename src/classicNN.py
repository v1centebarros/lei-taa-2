import os
import numpy as np
from sklearn.model_selection import train_test_split
from audioProcessor import AudioProcessor, Audio
import matplotlib.pyplot as plt
import seaborn as sns


from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing
from keras.optimizers import Adam


def main():
    # load train, cv and test data from npy files in folders and read the labels from the file name
    train_X = np.array([np.load(f"data/spectograms/melspectrogram/train/{file}") for file in os.listdir("data/spectograms/melspectrogram/train")])
    train_Y = np.array([int(file.split("_")[1]) for file in os.listdir("data/spectograms/melspectrogram/train")])
    cv_X = np.array([np.load(f"data/spectograms/melspectrogram/cv/{file}") for file in os.listdir("data/spectograms/melspectrogram/cv")])
    cv_Y = np.array([int(file.split("_")[1]) for file in os.listdir("data/spectograms/melspectrogram/cv")])
    test_X = np.array([np.load(f"data/spectograms/melspectrogram/test/{file}") for file in os.listdir("data/spectograms/melspectrogram/test")])
    test_Y = np.array([int(file.split("_")[1]) for file in os.listdir("data/spectograms/melspectrogram/test")])

    # Normalize the data

    model = Sequential(
        [
            Input(shape=(*train_X[0].shape,1)),
            Resizing(32,32),
            Conv2D(32,3,activation="relu"),
            MaxPooling2D(),
            Dropout(0.5),
            Flatten(),
            Dense(128,activation="relu"),
            Dense(10)
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_X, train_Y, epochs=40, validation_data=(cv_X, cv_Y))

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)

    print(f"Test accuracy: {test_acc}")

    model.summary()
        
if __name__ == "__main__":
    main()