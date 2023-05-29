import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys

y, sr = librosa.load(sys.argv[1])
S = librosa.stft(y, hop_length=64)
M = librosa.feature.melspectrogram(y, sr=sr, hop_length=64)
F = librosa.mel_frequencies(128)
bins = np.arange(128)

y2, sr2 = librosa.load(sys.argv[2])
S2 = librosa.stft(y2, hop_length=64)

if False:
    #plt.figure(figsize=(5, 1))
    plt.imshow(librosa.amplitude_to_db(np.abs(S), ref=np.max), aspect="auto")
    axes = plt.gca()
    axes.invert_yaxis()
    plt.title('')
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    #plt.yticks(np.concatenate((bins[::16], [bins[-1]])), np.round(np.concatenate((F[::16], [F[-1]]))))
    plt.yticks(np.linspace(0, S.shape[0], 9), np.round(np.linspace(0, 11025, 9)))
    plt.show()

fig, axes = plt.subplots(2)
axes[0].imshow(librosa.amplitude_to_db(np.abs(S), ref=np.max), aspect="auto")
axes[0].invert_yaxis()
axes[0].set_yticks([], [])
axes[0].set_ylabel(sys.argv[1])
axes[1].imshow(librosa.amplitude_to_db(np.abs(S2), ref=np.max), aspect="auto")
axes[1].invert_yaxis()
axes[1].set_xlabel("Frame")
axes[1].set_ylabel(sys.argv[2])
axes[1].set_yticks([], [])
plt.show()