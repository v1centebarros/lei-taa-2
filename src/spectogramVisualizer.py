import argparse

import numpy as np
from audioProcessor import AudioProcessor


def visualize_spectogram(spect_file):
    ap = AudioProcessor()
    spectogram = np.load(spect_file)
    ap.plot_spectogram(spectogram)



if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Visualize spectogram")
    argParser.add_argument("--path", type=str, help="Type of spectogram to use", required=True)
    args = argParser.parse_args()
    visualize_spectogram(args.path)
    