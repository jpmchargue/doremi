from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math


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

    print(f"{best_j}: total power = {best_power}")
    return best_j, best_power

def sweep_pitch_algorithm(y, sr, debug=False):
    m = np.max(np.abs(y))
    N = round(sr/5) # 5Hz fidelity
    overlap = 0.05 # 10 ms time precision
    windows, starts = get_hann_windows(y, N, R=overlap)

    min_hz = 75
    max_hz = 500

    bin_hz = (sr / N)
    min_bin = math.floor(min_hz / bin_hz)
    max_bin = math.ceil(max_hz / bin_hz)
    cutoff_hz = 1000
    cutoff_bin = math.ceil(cutoff_hz / bin_hz) + 1
    all_pitches = []
    all_powers = []
    for w, window in enumerate(windows):
        fft = np.fft.fft(window)
        board = np.abs(fft[:cutoff_bin])

        sweep_interval, sweep_power = sweep_frequencies(board, 5, cutoff_bin)

        if debug and w * (sr/(N*overlap)) > 0.38:
            plt.plot(board)
            m = np.average(board)
            plt.plot(m * np.ones(len(board)))
            i = sweep_interval
            while i < len(board) and np.interp(i, np.arange(len(board)), board) > m:
                plt.plot(i, np.interp(i, np.arange(len(board)), board), marker='o')
                i += sweep_interval
            plt.show()

        all_pitches.append(sweep_interval * bin_hz)
        all_powers.append(sweep_power)
    
    return (N/2) / sr, (N * overlap) / sr, all_pitches, all_powers/np.max(all_powers)

def get_spectrogram(y):
    N = round(sr/5) # 5Hz fidelity
    overlap = 0.05 # 10 ms time precision
    windows, starts = get_hann_windows(y, N, R=overlap)

    for window in windows:
        fft = np.fft.fft(window)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        fig, axes = plt.subplots(2)
        axes[0].plot(magnitude, color='b')
        axes[1].plot(phase, color='g')
        plt.show()

sr, y = wavfile.read("sample.wav")

poffset, phop, pitches, powers = sweep_pitch_algorithm(y, sr)
fig, axes = plt.subplots(3)
axes[0].plot(y, color='k')
axes[1].plot(pitches, color='b')
axes[2].plot(powers, color='y')
plt.show()

