from audioProcessor import AudioProcessor


def main():
    ap = AudioProcessor()

    for audio in ap.audio_files:
        # ap.plot_spectogram(ap.spectogram(audio))
        # ap.plot_spectogram(ap.spectogram(audio, spec_func="mfcc"))
        # ap.plot_spectogram(ap.spectogram(audio, spec_func="chroma_stft"))
        # ap.plot_waveform(audio)
        new_audio = ap.data_augmentation(audio, noise=False, pitch=True, speed=False)
        ap.plot_waveform(new_audio)
        ap.save_audio(new_audio, "test.wav")
        break


if __name__ == "__main__":
    main()