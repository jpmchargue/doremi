import numpy as np

from .util import (
    Signal, 
    Frames, 
    PitchMarkers
)
from .tuners import (
    sweep_pitch_algorithm, 
    get_pitch_markers, 
    tune_pitches, 
    smooth_pitches
)
from .td_psola import td_psola


def tune(filename, output_filename, key="C", scale="major", debug=False):
    signal = Signal.from_file(filename)

    frames = Frames(round(signal.sr/10), 0.8)
    frames.frame_signal(signal)

    pitches, energies = sweep_pitch_algorithm(frames)
    energy_range = max(energies.fx) - min(energies.fx)
    energy_threshold = (energy_range * 0.1) + min(energies.fx)

    source_markers = get_pitch_markers(frames, pitches)
    source_voiced_idx = np.where(energies[frames.get_frame_index(source_markers.markers)] > energy_threshold)
    analysis_markers = PitchMarkers(source_markers.markers[source_voiced_idx], source_markers.frequencies[source_voiced_idx])

    target_pitches = tune_pitches(pitches, tonic=key, scale=scale)
    target_pitches_smooth = smooth_pitches(target_pitches)
    target_markers = get_pitch_markers(frames, target_pitches_smooth)
    target_voiced = energies[frames.get_frame_index(target_markers.markers)] > energy_threshold

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(pitches.fx, label="Original")
        plt.plot(target_pitches_smooth.fx, color="blueviolet", label="Target")
        plt.title("Original and Target Frequencies")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Frame")
        plt.legend()
        plt.show()

    signal_output = td_psola(signal, analysis_markers, target_markers, target_voiced)

    signal_output.save_to(output_filename)