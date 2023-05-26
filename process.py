import numpy as np
from scipy import signal
import math
import soundfile as sf

class Signal:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
    
    def from_file(filename):
        y, sr = sf.read(filename)
        return Signal(y, sr)

class LerpArray:
    # A linearly interpolatable array.
    def __init__(self, array):
        self.fx = array
        self.x = np.arange(len(self.fx))
    
    def __getitem__(self, index):
        return np.interp(index, self.x, self.fx)

    def __len__(self):
        return len(self.fx)

class Frames:
    def __init__(self, window_length, overlap):
        self.window_length = window_length
        self.hop_length = math.ceil(window_length * (1 - overlap))
        self.hanning = signal.hann(self.window_length)

        self.signal = None
        self.sr = None
        self.frames = None
        self.centers = None

    def frame_signal(self, signal, sr):
        starts = np.arange(0, len(signal) - self.window_length + 1, self.hop_length)
        slices = np.array([signal[i:i+self.window_length] for i in starts])
        norms = slices - np.expand_dims(np.mean(slices, axis=1), axis=1)

        self.signal = signal
        self.sr = sr
        self.frames = norms * self.hanning
        self.centers = (self.window_length // 2) + (np.arange(len(starts)) * self.hop_length)

    def get_frame_index(self, sample_idx):
        return np.clip(
            (sample_idx - (self.window_length // 2)) / self.hop_length, 
            0, 
            len(self.frames) - 1
        )

    def get_closest_frame_index(self, sample_idx):
        return round(self.get_frame_index(sample_idx))

    def __len__(self):
        if self.frames is None:
            return 0
        return len(self.frames)


def sweep_pitch_algorithm(frames, precision=1.0):
    # returns a LerpArray of pitch frequencies and a LerpArray of energies for each frame
    f_min, f_max = 50, 1000
    bins_hz = frames.sr / frames.window_length
    bins_min, bins_max = math.floor(f_min / bins_hz), math.ceil(f_max / bins_hz)
    bins_precision = precision / bins_hz

    def sweep_frequencies(spectrum):
        bins_stop = min(bins_max, len(spectrum))
        threshold = np.average(spectrum.fx)
        best_bins, best_power = bins_min, 0
        for j in np.arange(bins_min, bins_stop, bins_precision):
            total_power = 0
            pointer = j
            while pointer < len(spectrum) and spectrum[pointer] > threshold:
                total_power += spectrum[pointer]
                pointer += j
            if total_power > best_power:
                best_bins, best_power = j, total_power

        best_frequency = best_bins * bins_hz
        print(f"Best Frequency: {best_frequency}")
        return best_frequency, best_power

    pitches, energies = [], []
    for frame in frames.frames:
        spectrum = LerpArray(np.abs(np.fft.rfft(frame)))
        p, e = sweep_frequencies(spectrum)
        pitches.append(p)
        energies.append(e)

    return LerpArray(pitches), LerpArray(energies)


def get_pitch_markers(frames, frame_pitches):
    pitch_markers = []
    i = 0
    while i < len(frames.signal):
        f_i = frame_pitches[frames.get_frame_index(i)]
        hop = round(frames.sr / f_i)
        i += hop
        pitch_markers.append(i)
    return pitch_markers


def get_window_from_signal(signal, shape):
    start_tail_length, start, end, end_tail_length = shape
    #hanning =

import matplotlib.pyplot as plt

y, sr = sf.read('sample.wav')

frames = Frames(round(sr/5), 0.95)
frames.frame_signal(y, sr)

pitches, energies = sweep_pitch_algorithm(frames)

print(len(frames))
print(len(pitches))
print(len(energies))

pitch_markers = get_pitch_markers(frames, pitches)
print(len(pitch_markers))

for test_i in range(50, 1000):
    marker_sample = np.array(pitch_markers[test_i:test_i+10])
    snip = np.array(y[marker_sample[0]:marker_sample[-1]])
    xs = marker_sample[:-1] - pitch_markers[test_i]
    print(xs)
    print(snip[xs])
    plt.plot(snip, color='b')
    plt.scatter(xs, snip[xs], 20, 'r', 'o')
    plt.show()

#fig, axes = plt.subplots(2)
#axes[0].plot(pitches.fx, color='b')
#axes[1].plot(energies.fx, color='y')
plt.show()


#sf.write('new_file.flac', data, samplerate)