import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys
import os

if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        y, sr = librosa.load(sys.argv[1])
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=64)), ref=np.max)
        axes = plt.gca()
        fig = axes.get_figure()
        fig.set_size_inches(6, 6)

        limit = 1024
        if len(sys.argv) > 2:
            proportion = np.clip(float(sys.argv[2]), 0, 1)
            limit = int(proportion * 1024)

        axes.imshow(S[:limit], interpolation='nearest', aspect='auto')
        axes.invert_yaxis()
        plt.show()
    else:
        print("ERROR: desired path does not exist")
else:
    print("ERROR: missing path argument")

