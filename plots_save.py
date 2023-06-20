import librosa
from matplotlib import pyplot as plt
from src.audioProcessor import AudioProcessor, Audio

data = AudioProcessor()

longest_audio = data.get_longest_audio()
data.plot_waveform(longest_audio)


                     
