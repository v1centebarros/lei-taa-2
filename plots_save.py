import librosa
from matplotlib import pyplot as plt
from src.audioProcessor import AudioProcessor, Audio

data = AudioProcessor()

longest_audio = data.get_longest_audio()
data.plot_waveform(longest_audio,title="Longest audio signal")

# remove silence
longest_audio = data.remove_silence(longest_audio)

# plot waveform
data.plot_waveform(longest_audio, title = "Longest audio signal after trimming")

# add padding
longest_audio = data.padding(longest_audio, len(longest_audio)+1000)

# plot waveform
data.plot_waveform(longest_audio, title="Longest audio signal after trimming and add padding")

# melspectrogram
spectogram = data.spectogram(longest_audio,"melspectrogram", power_to_db=True)
data.plot_spectogram(spectogram, "Mel-frequency Spectrogram")

# mfcc
spectogram = data.spectogram(longest_audio, "mfcc", power_to_db=False)
data.plot_spectogram(spectogram, "MFCC Spectrogram")

# chroma
spectogram = data.spectogram(longest_audio,"chroma_stft", power_to_db=True)
data.plot_spectogram(spectogram, "chroma_stft")

# stft
longest_audio = data.remove_silence(longest_audio)
spectogram = data.spectogram(longest_audio,"stft", power_to_db=True)
data.plot_spectogram(spectogram, "stft")

                     
# --------------------results models ---------------------------

# load file .h5
filename = "model_1_chroma_stft.h5"


