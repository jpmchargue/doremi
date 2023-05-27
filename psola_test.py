import librosa
import psola
import numpy as np
import soundfile as sf

y, sr = librosa.load("sample.wav")
target = np.array([300 * (2**(1/6))])

tuned = psola.vocode(y, sample_rate=int(sr), target_pitch=target, fmin=50, fmax=4200)
sf.write("tuned4.wav", tuned, sr)