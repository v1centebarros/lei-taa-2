import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from audioProcessor import AudioProcessor, Audio
import matplotlib.pyplot as plt
import seaborn as sns


from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy


def main():
    spectogram_type = "mfcc"

    # load train, cv and test data from npy files in folders and read the labels from the file name
    train_X = np.array([np.load(f"data/spectograms/{spectogram_type}/train/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    train_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    cv_X = np.array([np.load(f"data/spectograms/{spectogram_type}/cv/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    cv_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    test_X = np.array([np.load(f"data/spectograms/{spectogram_type}/test/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])
    test_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])

    # Normalize the data

    model = Sequential(
        [
            Input(shape=(*train_X[0].shape,1)),
            Resizing(32, 32), 
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            #layers.Conv2D(64, 3, activation='relu'),
            #layers.MaxPooling2D(),
            Dropout(0.5),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10),
        ]
    )
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    history = model.fit(train_X, train_Y, epochs=25, validation_data=(cv_X, cv_Y))

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)

    print(f"Test accuracy: {test_acc}")

    model.summary()

    metrics = history.history


    fig, ax = plt.subplots(1, 2)

    ax[0].plot(history.epoch, metrics['loss'], metrics['val_loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].legend(['train loss', 'validation loss'])


    ax[1].plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    #plt.plot([x + 25 for x in history.epoch], metrics['accuracy'])
    #plt.plot([x + 25 for x in history.epoch], metrics['val_accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylim([0, 1])
    ax[1].legend(['train accuracy', 'validation accuracy'])

    fig.set_size_inches(8, 5)
    plt.show()

    # Predict the test data
    y_pred = model.predict(test_X)
    predicted_categories = np.argmax(y_pred, axis=1)
    true_categories = test_Y

    # Plot a confusion matrix
    cm = confusion_matrix(true_categories, predicted_categories)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
        
if __name__ == "__main__":
    main()