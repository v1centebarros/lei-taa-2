from sklearn.model_selection import train_test_split
from audioProcessor import Audio, AudioProcessor
import os


def main():
    ap = AudioProcessor()

    sanitized_audio_data, _ = ap.sanitize_audio()
    spectogram_type = "mfcc"

    #Clean previous data
    for file in os.listdir(f"data/spectograms/{spectogram_type}/train"):
        os.remove(f"data/spectograms/{spectogram_type}/train/{file}")

    for file in os.listdir(f"data/spectograms/{spectogram_type}/test"):
        os.remove(f"data/spectograms/{spectogram_type}/test/{file}")

    for file in os.listdir(f"data/spectograms/{spectogram_type}/cv"):
        os.remove(f"data/spectograms/{spectogram_type}/cv/{file}")


    #Convert sanitized_audio_data to spectrogram
    sanitized_audio_data = [Audio(ap.spectogram(audio.sample,spec_func=spectogram_type),audio.label) for audio in sanitized_audio_data]

    #Split between train and test and cross validation

    train, test = train_test_split(sanitized_audio_data, test_size=0.4, random_state=42,stratify=[audio.label for audio in sanitized_audio_data])
    train, cv = train_test_split(train, test_size=0.5, random_state=42,stratify=[audio.label for audio in train])

    # Save the data in the respective folders
    for idx, audio in enumerate(train):
        ap.save_spectogram(audio.sample, f"label_{audio.label}_sample_{idx}",data_split="train",spectogram_type=spectogram_type)
    for idx, audio in enumerate(test):
        ap.save_spectogram(audio.sample, f"label_{audio.label}_sample_{idx}",data_split="test",spectogram_type=spectogram_type)
    for idx,audio in enumerate(cv):
        ap.save_spectogram(audio.sample,f"label_{audio.label}_sample_{idx}",data_split="cv",spectogram_type=spectogram_type)
        

if __name__ == "__main__":
    main()