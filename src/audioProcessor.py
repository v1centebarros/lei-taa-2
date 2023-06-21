import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import logging
import soundfile as sf
import seaborn as sns
from collections import namedtuple
from os.path import join
from scipy.io import wavfile

Audio = namedtuple("Audio", ["sample", "label"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spectogram_func = {
    "melspectrogram": librosa.feature.melspectrogram,
    "mfcc": librosa.feature.mfcc,
    "chroma_stft": librosa.feature.chroma_stft,
    "stft": librosa.stft,
}


class AudioProcessor:
    def __init__(
        self,
        sample_rate=8000,
        audio_folder="data/recordings",
        output_folder="data/output",
        spectograms_folder="data/spectograms",
    ):
        self.sample_rate = sample_rate
        self.audio_folder = audio_folder
        self.output_folder = output_folder
        self.spectograms_folder = spectograms_folder
        self.audio_files:list[Audio] = self.load_audio_files()

    def spectogram(self, waveform = None, spec_func="mfcc", sr=8000,power_to_db=False):

        if waveform is None:
            logger.info("No waveform passed, using the longest audio")
            waveform = self.get_longest_audio()

        # Convert to spectrogram
        logger.info(f"Converting to {spec_func} spectrogram")
        spec = spectogram_func[spec_func](y=waveform, sr=sr)
        
        if power_to_db:
            return librosa.power_to_db(spec, ref=np.max)
        
        return spec
    
    def plot_spectogram(self, spec, title="Mel spectrogram"):
        logger.info("Plotting Spectogram")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec, y_axis="mel", x_axis="time",sr=self.sample_rate)
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        filename = title.replace(" ", '')
        plt.savefig(join(self.output_folder, filename + ".png"))
        plt.close()

    def spectogram_stft(self, waveform = None, spec_func="stft", sr=8000,power_to_db=False):
            
        # Padding for files with less than 16000 samples
        if waveform is None:
            logger.info("No waveform passed, using the longest audio")
            waveform = self.get_longest_audio()

        # Convert to spectrogram
        spec = librosa.stft(waveform)
        # Log-scale frequency axis so that both positive and negative frequencies can be clearly see
        spec_db = librosa.amplitude_to_db(np.abs(spec))

        return spec_db

    def plot_waveform(self, waveform, title="Waveform"):
        logger.info("Plotting waveform")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(waveform, sr=self.sample_rate)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.tight_layout()
        filename = title.replace(" ", '')
        plt.savefig(join(self.output_folder, filename + ".png"))
        plt.close()


    def load_audio_files(self, audio_folder=None) -> list[Audio]:
        if audio_folder is None:
            audio_folder = self.audio_folder

        audio_files = librosa.util.find_files(audio_folder, ext=["wav"])
        logger.info(f"Found {len(audio_files)} audio files in {audio_folder}")

        return [Audio(librosa.load(file, sr=self.sample_rate)[0], file.split("/")[-1][0]) 
                       for file in audio_files]

    def remove_silence(self, sample, top_db=10):
        # Trim the beginning and ending silence
        logger.info("Removing silence")
        audio, _ = librosa.effects.trim(sample, top_db=top_db)
        return audio

    def padding(self, sample, length):
        logger.info("Padding the sample")
        return librosa.util.fix_length(sample,size=length)

    def add_noise(self, sample, noise_factor=0.05):
        logger.info("Adding noise")
        noise = np.random.randn(len(sample))
        augmented_sample = sample + noise_factor * noise
        augmented_sample = augmented_sample.astype(type(sample[0]))

        return Audio(augmented_sample, sample.label)

    def change_pitch(self, sample):
        logger.info("Changing pitch")
        return librosa.effects.pitch_shift(sample, sr=self.sample_rate, n_steps=4)

    def change_speed(self, sample, speed_factor=0.7):
        logger.info("Changing speed")
        return librosa.effects.time_stretch(sample, rate=speed_factor)

    def data_augmentation(self, audio, noise=False, pitch=False, speed=False):
        sample = audio.sample
        if noise:
            sample = self.add_noise(sample)
        if pitch:
            sample = self.change_pitch(sample)
        if speed:
            sample = self.change_speed(sample)
        return Audio(sample, audio.label)

    def save_audio(self, sample, filename):
        sf.write(os.path.join(self.output_folder, filename), sample, self.sample_rate)
        logger.info(f"Audio saved in {os.path.join(self.output_folder,filename)}")

    def save_spectogram(self, spec, filename,spectogram_type="melspectrogram",data_split="train"):
        #Save the spectrogram as a pickle file
        logger.info(f"Saving spectrogram in {os.path.join(self.spectograms_folder,spectogram_type,data_split,filename)}")
        np.save(os.path.join(self.spectograms_folder,spectogram_type,data_split,filename), spec)

        
    def get_longest_audio(self, audio_files=None):
        if audio_files is None:
            logger.info("Getting longest audio from default audio files")
            audio_files = self.audio_files

        return max(audio_files, key=lambda x: len(x.sample)).sample

    
    def sanitize_audio(self, samples= None):

        if samples is None:
            logger.info("Sanitizing default audio files")
            samples = self.audio_files
        
        logger.info("Trimming audio")
        trimmed_audio = [Audio(self.remove_silence(sample.sample),sample.label) for sample in samples]
        

        longest_audio = self.get_longest_audio(trimmed_audio)
        logger.info(f"Longest audio has {len(longest_audio)} samples")

        logger.info("Padding audio")
        padded_audio = [Audio(self.padding(sample.sample, len(longest_audio)),sample.label) for sample in trimmed_audio]

        return padded_audio, longest_audio
        
    
    def show_length_distribution(self,signals):
        plt.rcParams["figure.titleweight"] = 'bold' 
        plt.rcParams["figure.titlesize"] = 'large'
        plt.rcParams['figure.dpi'] = 120
        sampel_times = [len(x)/self.sample_rate for x in signals]


        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

        # Add a graph in each part
        sns.boxplot(x = sampel_times, ax=ax_box, linewidth = 0.9, color=  '#9af772')
        sns.histplot(x = sampel_times, ax=ax_hist, bins = 'fd', kde = True)

        # Remove x axis name for the boxplot
        ax_box.set(xlabel='')


        title = 'Audio signal lengths'
        x_label = 'duration (seconds)'
        y_label = 'count'

        plt.suptitle(title)
        ax_hist.set_xlabel(x_label)
        ax_hist.set_ylabel(y_label)
        plt.show()

        return sampel_times

    
    