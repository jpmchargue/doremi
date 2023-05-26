# get pitches
# determine freq factors for each frame 
# pass through windowing
# for each window:
    # get frequency content
    # transform all frequencies by freq factor
    # construct new frame manually from transformed frequencies
    # window new frame
    # take fft of new window
    # EITHER:
        # adjust phase based on frame hop, OR
        # try resynthesis pull method

from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import soundfile as sf
from tqdm import tqdm

def get_hann_windows(y, N=240, R=0.5):
    hop = math.ceil(N * R)
    hann_window = signal.hann(N)
    starts = [i for i in range(0, len(y) - N + 1, hop)]
    slices = [y[i:i+N] for i in starts]
    norms = [slice - np.average(slice) for slice in slices]
    return np.array([norm * hann_window for norm in norms]), np.array(starts)

def sweep_frequencies(fft, low_bins=5, high_bins=50, precision=0.1):
    # Sweep through all possible frequency intervals 
    # to find the one that best fits the observed FFT.
    # Starting at j = low_bins, jump through the FFT at intervals of j
    # and add the power at each frequency seen until the power is below
    # an arbitrary 'not present' threshold (e.g. 0.1).
    # Iterate through all values of j from low_bins to high_bins.
    # Return the value of j with the highest overall harmonic power.
    xs = np.arange(len(fft))
    threshold = np.average(fft)
    best_power, best_j = 0, low_bins
    for j in np.arange(low_bins, high_bins, precision):
        total_power = 0
        pointer = j
        while pointer < len(fft) and np.interp(pointer, xs, fft) > threshold:
            total_power += np.interp(pointer, xs, fft)
            pointer += j
        if total_power > best_power:
            best_power, best_j = total_power, j
        #print(f"{j}: total power = {total_power}")

    #print(f"{best_j}: total power = {best_power}")
    return best_j, best_power

def get_freq_factor(freq):
    root = np.power(2, 1/12)
    target_freq = np.power(root, round(np.log(freq/55)/np.log(root))) * 55
    return freq / target_freq

def overlap_add(base, segment, offset, hann=False):
    if hann:
        segment = segment * signal.hann(len(segment))
    if len(base) == 0:
        base.extend(segment)
    else:
        for i in range(len(segment) - offset):
            base[-i - 1] += segment[len(segment) - offset - 1 - i]
        for i in range(offset):
            base.append(segment[-i])

sr, y = wavfile.read("sample.wav")
N = round(sr/5)
overlap = 0.05
hop_length = int(np.ceil(N * overlap))

bin_hz = (sr / N)
cutoff_hz = 1000
cutoff_bin = int(np.floor(cutoff_hz / bin_hz) + 1)

base = []
windows, starts = get_hann_windows(y, N, R=overlap)
for window in tqdm(windows):
    #plt.plot(window)
    #plt.show()

    rfft = np.fft.rfft(window)
    sweep_interval, sweep_power = sweep_frequencies(rfft, 5, 200)
    freq_factor = get_freq_factor(sweep_interval * bin_hz)
    tqdm.write(f"Freq Factor: {freq_factor}")

    abs_rfft = np.abs(rfft)
    abs_rfft[0] = 0
    xs = np.arange(len(rfft))
    new_rfft = np.zeros(len(rfft)).astype("complex")
    for i in range(1, len(rfft)):
        mag = np.interp(i * freq_factor, xs, abs_rfft)
        phase = np.angle(rfft[i])
        new_rfft[i] = (mag * np.cos(phase)) + (mag * 1j * np.sin(phase))
    new_rfft[0] = rfft[0]

    #fig, axes = plt.subplots(2)
    #axes[0].plot(abs_rfft[:cutoff_bin])
    #axes[1].plot(np.abs(new_rfft)[:cutoff_bin])
    #plt.show()

    new_window = np.fft.irfft(new_rfft)

    #plt.plot(new_window)
    #plt.show()

    overlap_add(base, new_window, hop_length)

sf.write("output.wav", base, sr)
    