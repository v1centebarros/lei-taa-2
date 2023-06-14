from sklearn.model_selection import train_test_split
from audioProcessor import Audio, AudioProcessor


def main():
    ap = AudioProcessor()

    sanitized_audio_data, _ = ap.sanitize_audio()

    #Convert sanitized_audio_data to spectrogram
    sanitized_audio_data = [Audio(ap.spectogram(audio.sample),audio.label) for audio in sanitized_audio_data]

    #Split between train and test and cross validation

    train, test = train_test_split(sanitized_audio_data, test_size=0.2, random_state=42,stratify=[audio.label for audio in sanitized_audio_data])
    train, cv = train_test_split(train, test_size=0.2, random_state=42,stratify=[audio.label for audio in train])

    # Save the data in the respective folders
    for idx, audio in enumerate(train):
        ap.save_spectogram(audio.sample, f"label_{audio.label}_sample_{idx}",data_split="train")
    for idx, audio in enumerate(test):
        ap.save_spectogram(audio.sample, f"label_{audio.label}_sample_{idx}",data_split="test")
    for idx,audio in enumerate(cv):
        ap.save_spectogram(audio.sample,f"label_{audio.label}_sample_{idx}",data_split="cv")
        

if __name__ == "__main__":
    main()