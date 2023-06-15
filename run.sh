#!/bin/bash

#Load Spectogram
python3 src/dataProcess.py --spectogram_type melspectrogram --power_to_db False

python3 src/dataProcess.py --spectogram_type mfcc --power_to_db False

python3 src/dataProcess.py --spectogram_type chroma_stft --power_to_db False

python3 src/modelGym.py

mv data/models/models_performance.json data/models/models_performance_power_to_db_false.json

python3 src/dataProcess.py --spectogram_type melspectrogram --power_to_db True

python3 src/dataProcess.py --spectogram_type mfcc --power_to_db True

python3 src/dataProcess.py --spectogram_type chroma_stft --power_to_db True

python3 src/modelGym.py

mv data/models/models_performance.json data/models/models_performance_power_to_db_true.json
