import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import logging
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spectogram_func = {
    "melspectrogram": librosa.feature.melspectrogram,
    "mfcc": librosa.feature.mfcc,
    "chroma_stft": librosa.feature.chroma_stft,
}


class AudioProcessor:
    def __init__(
        self,
        sample_rate=8000,
        audio_folder="data/recordings",
        output_folder="data/output",
    ):
        self.sample_rate = sample_rate
        self.audio_folder = audio_folder
        self.output_folder = output_folder
        self.audio_files = self.load_audio_files()

    def spectogram(self, waveform, spec_func="melspectrogram", sr=8000):
        # Convert to spectrogram
        spec = spectogram_func[spec_func](y=waveform, sr=sr)
        return librosa.power_to_db(spec, ref=np.max)

    # TODO: Refactor this method to handle different name
    def plot_spectogram(self, spec):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec, y_axis="mel", fmax=8000, x_axis="time")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram")
        plt.tight_layout()
        plt.show()

    def plot_waveform(self, waveform):
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(waveform, sr=self.sample_rate)
        plt.show()

    def load_audio_files(self, audio_folder=None):
        if audio_folder is None:
            audio_folder = self.audio_folder

        audio_files = librosa.util.find_files(audio_folder, ext=["wav"])
        logger.info(f"Found {len(audio_files)} audio files in {audio_folder}")

        return [librosa.load(file, sr=self.sample_rate)[0] for file in audio_files]

    def remove_silence(self, sample, top_db=10):
        # Trim the beginning and ending silence
        audio, _ = librosa.effects.trim(sample, top_db=top_db)
        return audio

    def padd_truncate(self, sample, length=8000):
        # TODO: Implement this method if needed
        raise NotImplementedError

    def add_noise(self, sample, noise_factor=0.05):
        noise = np.random.randn(len(sample))
        augmented_sample = sample + noise_factor * noise
        augmented_sample = augmented_sample.astype(type(sample[0]))

        return augmented_sample

    def change_pitch(self, sample):
        return librosa.effects.pitch_shift(sample, sr=self.sample_rate, n_steps=4)

    def change_speed(self, sample, speed_factor=0.7):
        return librosa.effects.time_stretch(sample, rate=speed_factor)

    def data_augmentation(self, sample, noise=False, pitch=False, speed=False):
        if noise:
            logger.info("Adding noise to sample")
            sample = self.add_noise(sample)
        if pitch:
            logger.info("Changing pitch of sample")
            sample = self.change_pitch(sample)
        if speed:
            logger.info("Changing speed of sample")
            sample = self.change_speed(sample)
        return sample

    def save_audio(self, sample, filename):
        sf.write(os.path.join(self.output_folder, filename), sample, self.sample_rate)
        logger.info(f"Audio saved in {os.path.join(self.output_folder,filename)}")
        