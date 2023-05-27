import numpy as np
from scipy import signal
import math
import soundfile as sf
from tqdm import tqdm

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
        self.fx = np.array(array)
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
        print(f"Best Frequency: {round(best_frequency, 5)}")
        return best_frequency, best_power

    pitches, energies = [], []
    for frame in frames.frames:
        spectrum = LerpArray(np.abs(np.fft.rfft(frame)))
        p, e = sweep_frequencies(spectrum)
        pitches.append(p)
        energies.append(e)

    return LerpArray(pitches), LerpArray(energies)


class PitchMarkers:
    def __init__(self, markers, markers_f):
        self.markers = np.array(markers)
        self.frequencies = np.array(markers_f)

    def __len__(self):
        return len(self.markers)

def get_pitch_markers(frames, frame_pitches):
    pitch_markers, pitch_markers_f = [], []
    i = 0
    while i < len(frames.signal):
        f_i = frame_pitches[frames.get_frame_index(i)]
        hop = round(frames.sr / f_i)
        i += hop
        pitch_markers.append(i)
        pitch_markers_f.append(f_i)
    return PitchMarkers(pitch_markers, pitch_markers_f)


class Window:
    def __init__(self, y):
        self.weights = signal.hann(len(y))
        self.window = self.weights * y

def tune_pitches(pitches):
    twelfth_root_two = 2**(1/6)
    return LerpArray((1.5) * 440.0*(twelfth_root_two**np.round(np.log(pitches.fx/440.0)/np.log(twelfth_root_two))))

def set_pitches(pitches, f=220):
    return LerpArray(np.ones(len(pitches)) * f)


def td_psola(y, sr, analysis_markers, target_markers, voiced=None):
    y_output, y_density = np.zeros(len(y)), np.zeros(len(y)) + 0.00001

    for i in tqdm(range(len(target_markers))):
        target_idx = target_markers.markers[i]
        closest_analysis_marker = np.argmin(np.abs(target_idx - analysis_markers.markers))
        #analysis_window = get_analysis_window(closest_analysis_marker)

        analysis_idx = analysis_markers.markers[closest_analysis_marker]
        analysis_f = analysis_markers.frequencies[closest_analysis_marker]
        period = round(sr / analysis_f)
        t_s, t_e = analysis_idx - (2*period), analysis_idx + (2*period)
        #if voiced is None or voiced[closest_analysis_marker]:
        t_sa, t_ea = int(target_idx - (2*period)), int(target_idx + (2*period))
        #else:
        #    t_sa, t_ea = t_s, t_e

        if t_s < 0 or t_sa < 0:
            continue
        if t_e >= len(y) or t_ea > len(y):
            break

        window = Window(y[t_s:t_e])
        y_output[t_sa:t_ea] += window.window
        y_density[t_sa:t_ea] += window.weights

    #return y_output / np.max(np.abs(y_output)) #/ y_density
    #return y_output / y_density
    return y_output

        


#def get_window_from_signal(signal, shape):
#    start_tail_length, start, end, end_tail_length = shape
#    hanning =

import matplotlib.pyplot as plt
debug_pitch_synchronicity = False

y, sr = sf.read('sample.wav')

frames = Frames(round(sr/5), 0.95)
frames.frame_signal(y, sr)

pitches, energies = sweep_pitch_algorithm(frames)
energy_threshold = np.average(energies.fx) * 0.2

print(len(frames))
print(len(pitches))
print(len(energies))

pitch_markers = get_pitch_markers(frames, pitches)
print(len(pitch_markers))

if debug_pitch_synchronicity: # DEBUG
    for test_i in range(50, 1000):
        marker_sample = np.array(pitch_markers[test_i:test_i+10])
        snip = np.array(y[marker_sample[0]:marker_sample[-1]])
        xs = marker_sample[:-1] - pitch_markers[test_i]
        print(xs)
        print(snip[xs])
        plt.plot(snip, color='b')
        plt.scatter(xs, snip[xs], 20, 'r', 'o')
        plt.show()

#target_pitches = set_pitches(pitches, f=300)
target_pitches = tune_pitches(pitches)
target_markers = get_pitch_markers(frames, target_pitches)

plt.plot([pitch_marker for pitch_marker in pitch_markers.markers], [energies[frames.get_frame_index(pitch_marker)] for pitch_marker in pitch_markers.markers], 'y')
plt.plot([pitch_marker for pitch_marker in pitch_markers.markers], [energy_threshold for pitch_marker in pitch_markers.markers], 'b')
plt.show()

voiced = [energies[frames.get_frame_index(pitch_marker)] > energy_threshold for pitch_marker in pitch_markers.markers]
#y_output = td_psola(y, sr, pitch_markers, target_markers)

import psola
y_psola = psola.vocode(y, sample_rate=int(sr), target_pitch=target_pitches.fx, fmin=50, fmax=4200)

#fig, axes = plt.subplots(2)
#axes[0].plot(pitches.fx, color='b')
#axes[1].plot(energies.fx, color='y')
#plt.show()


#sf.write('new_file9.wav', y_output, sr)
sf.write('new_file9_psola.wav', y_psola, sr)