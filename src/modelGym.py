import argparse
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from models import model_1,model_2,model_3,model_4, model_5



def load_data(spectogram_type):
    train_X = np.array([np.load(f"data/spectograms/{spectogram_type}/train/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    train_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/train")])
    cv_X = np.array([np.load(f"data/spectograms/{spectogram_type}/cv/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    cv_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/cv")])
    test_X = np.array([np.load(f"data/spectograms/{spectogram_type}/test/{file}") for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])
    test_Y = np.array([int(file.split("_")[1]) for file in os.listdir(f"data/spectograms/{spectogram_type}/test")])

    return train_X, train_Y, cv_X, cv_Y, test_X, test_Y


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



def run_model(model, train_X, train_Y, cv_X, cv_Y, test_X, test_Y, plot=False, save=False, model_name="model_1"):
    history = train_model(model, train_X, train_Y, cv_X, cv_Y)

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)

    print(f"Test accuracy: {test_acc}")

    model.summary()

    if plot:
        plot_accuracy_loss(history)
        # Predict the test data
        y_pred = model.predict(test_X)
        predicted_categories = np.argmax(y_pred, axis=1)
        true_categories = test_Y
        plot_confusion_matrix(true_categories, predicted_categories)

    if save:
        save_model(model, model_name)

    return test_loss, test_acc

def save_model(model, model_name):
    model.save(f"data/models/{model_name}.h5")




def run(spectogram_type="mfcc"):

    train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data(spectogram_type)

    models_to_run = [model_1, model_2, model_3, model_4, model_5]

    models_performance = {}

    for i, model in enumerate(models_to_run):   
        print(f"Running model {i+1}")
        loss, accuracy = run_model(model(train_X), train_X, train_Y, cv_X, cv_Y, test_X, test_Y, plot=False, save=True, model_name=f"model_{i+1}_{spectogram_type}")
        models_performance[f"model_{i+1}_{spectogram_type}"] = {"loss": loss, "accuracy": accuracy}

    return models_performance


    
        
if __name__ == "__main__":
    #argParser = argparse.ArgumentParser(description="Train a model")
    #argParser.add_argument("--spectogram_type", type=str, help="Type of spectogram to use", required=True,  choices=["mfcc","melspectrogram","chroma_stft"])
    #args = argParser.parse_args()
    #run(args.spectogram_type)

    models_performance = {}

    for spectogram_type in ["mfcc","melspectrogram","chroma_stft"]:
        models_performance[spectogram_type] = run(spectogram_type)


    # with open(f'data/models/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_models_performance.json', 'w') as fp:
    #     json.dump(models_performance, fp, indent=4)
        
    with open(f'data/models/models_performance.json', 'w') as fp:
        json.dump(models_performance, fp, indent=4)

    