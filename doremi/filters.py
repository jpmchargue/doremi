import numpy as np

from .util import Signal


def comb_filter(signal, delay, decay_factor):
    delay_samples = round(signal.sr * delay)
    signal_output = np.copy(signal.y)
    for i in range(delay_samples, len(signal_output)-1):
        signal_output[i] += decay_factor * signal_output[i-delay_samples]
    return Signal(signal_output, signal.sr)

def all_pass_filter(signal):
    ap_delay, ap_decay_factor = 0.09, 0.13
    ap_delay_samples = round(signal.sr * ap_delay)
    signal_output = np.copy(signal.y)
    for i in range(ap_delay_samples, len(signal_output)):
        signal_output[i] += \
                ap_decay_factor * signal_output[i+20-ap_delay_samples] - \
                ap_decay_factor * signal_output[i-ap_delay_samples]

    # Normalize the signal
    signal_output = signal_output / np.max(np.abs(signal_output))

    return Signal(signal_output, signal.sr)

def reverb(signal, delay=0.5, decay_factor=0.15, inner_mix=0.5, wet_mix=1.0):
    """
    Returns the same signal, with reverb applied.
    """
    delay_variation = [0, -0.01173, 0.01931, -0.00797]
    decay_factor_variation = [0, -0.1313, -0.2743, -0.31]
    comb_filter_signals = []
    for i in range(4):
        comb_filter_signal = comb_filter(signal, delay+delay_variation[i], decay_factor+decay_factor_variation[i])
        comb_filter_signals.append(comb_filter_signal.y)
    comb_filter_output = Signal(np.mean(comb_filter_signals, axis=0), signal.sr)

    inner_output = Signal((inner_mix * comb_filter_output.y) + ((1 - inner_mix) * signal.y), signal.sr) 

    all_pass_1 = all_pass_filter(inner_output)
    all_pass_2 = all_pass_filter(all_pass_1)

    signal_output = Signal((wet_mix * all_pass_2.y) + ((1 - wet_mix) * signal.y), signal.sr) 

    return signal_output

