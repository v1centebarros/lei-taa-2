from src.models import model_3
import matplotlib.pyplot as plt
from modelGym import train_model, load_data

# --------------------------------model3 ----------------
train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data("melspectrogram")
model = model_3(train_X)
history = train_model(model, train_X, train_Y, cv_X, cv_Y)
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
metrics_melspectrogram = history.history

train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data("mfcc")
model = model_3(train_X)
history = train_model(model, train_X, train_Y, cv_X, cv_Y)
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
metrics_mfcc = history.history

train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data("chroma_stft")
model = model_3(train_X)
history = train_model(model, train_X, train_Y, cv_X, cv_Y)
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
metrics_chroma_stft = history.history

train_X, train_Y, cv_X, cv_Y, test_X, test_Y = load_data("stft")
model = model_3(train_X)
history = train_model(model, train_X, train_Y, cv_X, cv_Y)
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
metrics_stft = history.history


plt.figure(figsize=(5, 4))
plt.plot(history.epoch, metrics_stft['val_accuracy'])
plt.plot(history.epoch, metrics_mfcc['val_accuracy'])
plt.plot(history.epoch, metrics_melspectrogram['val_accuracy'])
plt.plot(history.epoch, metrics_chroma_stft['val_accuracy'])
plt.xlabel('Epochs')
plt.legend(["stft","mfcc", "melspectrogram", "chroma_stft"])
plt.show()