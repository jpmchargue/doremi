import numpy as np
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
        self.hanning = hann(self.window_length)

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

def hann(n):
    # A gross, unreadable recreation of scipy.signal.hann(M).
    return 0.5*np.cos(2*np.pi*((np.arange(n)/(n-1))-0.5))+0.5

def sweep_pitch_algorithm(frames, precision=1.0):
    # returns a LerpArray of pitch frequencies and a LerpArray of energies for each frame
    f_min, f_max = 50, 1000
    bins_hz = frames.sr / frames.window_length
    bins_min, bins_max = math.floor(f_min / bins_hz), math.ceil(f_max / bins_hz) + 1
    bins_precision = precision / round(bins_hz)

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
        return best_frequency, best_power

    pitches, energies = [], []
    for frame in tqdm(frames.frames):
        board = np.abs(np.fft.fft(frame)[:bins_max])
        spectrum = LerpArray(board)
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
    def __init__(self, y, periods=None):
        if periods is None:
            self.weights = hann(len(y))
            self.window = self.weights * y
        else:
            period_left, period_right = periods
            self.weights = np.zeros(period_left + period_right)
            self.weights[:period_left] = hann(2 * period_left)[:period_left]
            self.weights[-period_right:] = hann(2 * period_right)[-period_right:]
            self.window = self.weights * y


def tune_pitches(pitches, tonic="C", scale='major'):
    tonic_frequencies = {
        "A": 440.0,
        "Bb": 466.16,
        "B": 493.88,
        "C": 523.25,
        "C#": 554.37,
        "D": 587.33,
        "D#": 622.25,
        "E": 659.25,
        "F": 698.46,
        "F#": 739.99,
        "G": 783.99,
        "G#": 830.61
    }
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11, 12],
        "minor": [0, 2, 3, 5, 7, 8, 10, 12],
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "pentatonic": [0, 2, 4, 7, 9, 12]
    }

    if type(tonic) == str:
        tonic_f = tonic_frequencies[tonic] if tonic in tonic_frequencies else 440.0
    else:
        tonic_f = tonic
    scale_tones = np.array(scales[scale])

    twelfth_root_two = 2**(1/12)
    log_pitches = np.log(pitches.fx/tonic_f)/np.log(twelfth_root_two)
    octave, tones = log_pitches // 12, log_pitches % 12
    closest_scale_tones = scale_tones[np.argmin(np.abs(np.expand_dims(tones, axis=1) - scale_tones), axis=1)]
    tuned_pitches = tonic_f * (twelfth_root_two**((octave * 12) + closest_scale_tones))

    return LerpArray(tuned_pitches)

def smooth_pitches(pitches, minimum_frames=4):
    """
    Smoothes the target pitches of an autotuned sound.
    If a target pitch is held for a segment of fewer than 'minimum_frames' frames,
    and the target pitches on either end of the segment are the same,
    then the segment is adjusted to match the target pitches on either end.
    This helps to avoid one-frame jumps in the pitch due to vibrato.
    Setting 'minimum_frames' to 1 disables smoothing.
    """
    # Create list of all segments
    segments = []
    segment_pitch, segment_duration = pitches.fx[0], 1
    for i in range(1, len(pitches.fx)):
        if pitches.fx[i] == segment_pitch:
            segment_duration += 1
        else:
            segments.append([segment_pitch, segment_duration])
            segment_pitch, segment_duration = pitches.fx[i], 1
    segments.append([segment_pitch, segment_duration])

    def smooth_segments_with_length(l):
        i = 1
        while i < len(segments) - 1:
            if segments[i][1] != l:
                i += 1
            elif segments[i-1][0] == segments[i+1][0]:
                segments[i-1][1] += (segments[i][1] + segments[i+1][1])
                segments.pop(i+1)
                segments.pop(i)
            else:
                i += 1

    # Order of operations matters, so smooth the smallest segments first
    for l in range(1, minimum_frames):
        smooth_segments_with_length(l)
    
    new_pitches = []
    for segment in segments:
        new_pitches += [segment[0]] * segment[1]

    return LerpArray(new_pitches)


def set_pitches(pitches, f=220):
    return LerpArray(np.ones(len(pitches)) * f)

def td_psola(y, sr, analysis_markers, target_markers, target_voiced=None):
    """
    An implementation of the Time Domain Pitch Synchronous Overlap-Add algorithm,
    with additional modifications inspiried by Praat.
    """
    y_output = np.zeros(len(y))

    for i in tqdm(range(len(target_markers))):
        target_idx = target_markers.markers[i]
        closest_analysis_marker = np.argmin(np.abs(target_idx - analysis_markers.markers))

        if target_voiced is not None and not target_voiced[i]:
            # If this target marker is in an unvoiced (non-tonal) segment of the source signal,
            # use the corresponding segment of the source sound directly
            analysis_idx = target_idx
            analysis_f = target_markers.frequencies[i]
            period_left = target_idx - target_markers.markers[i - 1] if i > 0 else target_idx
            period_right = target_markers.markers[i + 1] - target_idx if i < len(target_markers.markers) - 1 else round(sr / analysis_f)
        else:
            # Use a segment of the source sound from the nearest analysis marker
            analysis_idx = analysis_markers.markers[closest_analysis_marker]
            analysis_f = analysis_markers.frequencies[closest_analysis_marker]
            period_left = min(
                    analysis_idx - analysis_markers.markers[closest_analysis_marker - 1] if closest_analysis_marker > 0 else analysis_idx,
                    target_idx - target_markers.markers[i - 1] if i > 0 else target_idx
            )
            period_right = min(
                    analysis_markers.markers[closest_analysis_marker + 1] - analysis_idx if closest_analysis_marker < len(analysis_markers.markers) - 1 else round(sr / analysis_f),
                    target_markers.markers[i + 1] - target_idx if i < len(target_markers.markers) - 1 else round(sr / analysis_f)
            )

        t_s, t_e = analysis_idx - period_left, analysis_idx + period_right
        t_sa, t_ea = int(target_idx - period_left), int(target_idx + period_right)
        if t_s < 0 or t_sa < 0:
            continue
        if t_e >= len(y) or t_ea > len(y):
            break

        window = Window(y[t_s:t_e], (period_left, period_right))
        y_output[t_sa:t_ea] += window.window

    return y_output


import matplotlib.pyplot as plt
debug_autotune = False

y, sr = sf.read('examples/original.wav')

frames = Frames(round(sr/10), 0.8)
frames.frame_signal(y, sr)

pitches, energies = sweep_pitch_algorithm(frames)
#energy_threshold = np.average(energies.fx) * 0.2
energy_range = max(energies.fx) - min(energies.fx)
energy_threshold = (energy_range * 0.1) + min(energies.fx)

source_markers = get_pitch_markers(frames, pitches)
source_voiced_idx = np.where(energies[frames.get_frame_index(source_markers.markers)] > energy_threshold)
analysis_markers = PitchMarkers(source_markers.markers[source_voiced_idx], source_markers.frequencies[source_voiced_idx])

#target_pitches = set_pitches(pitches, f=300)
target_pitches = tune_pitches(pitches, tonic="D", scale="major")
target_pitches_smooth = smooth_pitches(target_pitches)
target_markers = get_pitch_markers(frames, target_pitches_smooth)
target_voiced = energies[frames.get_frame_index(target_markers.markers)] > energy_threshold

plt.plot(pitches.fx, label="Original")
plt.plot(target_pitches_smooth.fx, color="blueviolet", label="Target")
plt.title("Original and Target Frequencies")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Frame")
plt.legend()
plt.show()

fig, axes = plt.subplots(2)
axes[0].plot([pitch_marker for pitch_marker in source_markers.markers], [energies[frames.get_frame_index(pitch_marker)] for pitch_marker in source_markers.markers], 'y')
axes[0].plot([pitch_marker for pitch_marker in source_markers.markers], [energy_threshold for pitch_marker in source_markers.markers], 'b')
axes[1].plot(target_voiced * 1.0, 'k')
plt.show()

#voiced = [energies[frames.get_frame_index(pitch_marker)] > energy_threshold for pitch_marker in pitch_markers.markers]
y_output = td_psola(y, sr, analysis_markers, target_markers, target_voiced)

#import psola
#y_psola = psola.vocode(y, sample_rate=int(sr), target_pitch=target_pitches.fx, fmin=50, fmax=4200)

#fig, axes = plt.subplots(2)
#axes[0].plot(pitches.fx, color='b')
#axes[1].plot(energies.fx, color='y')
#plt.show()


#sf.write('new_file9.wav', y_output, sr)
sf.write('examples/Dmajor.wav', y_output, sr)