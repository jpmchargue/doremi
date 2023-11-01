import numpy as np

from .util import (
    Signal, 
    Frames, 
    PitchMarkers,
    Segments
)
from .tuners import (
    sweep_pitch_algorithm, 
    get_pitch_markers, 
    tune_pitches, 
    smooth_pitches,
    balance_pitches
)
from .td_psola import td_psola


def tune(filename, output_filename, key="C", scale="major", attack=0.0, strength=1.0, minimum_vib_frames=8, minimum_gliss_frames=8, debug=False):
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
    target_pitches_smooth = smooth_pitches(target_pitches, minimum_vib_frames=minimum_vib_frames, minimum_gliss_frames=minimum_gliss_frames)
    target_pitches_balanced = balance_pitches(pitches, target_pitches_smooth, attack=attack, strength=strength)
    target_markers = get_pitch_markers(frames, target_pitches_balanced)
    target_voiced = energies[frames.get_frame_index(target_markers.markers)] > energy_threshold

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(pitches.fx, label="Original")
        plt.plot(target_pitches_balanced.fx, color="blueviolet", label="Target")
        plt.title("Original and Target Frequencies")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Frame")
        plt.legend()
        plt.show()

    signal_output = td_psola(signal, analysis_markers, target_markers, target_voiced)

    if output_filename:
        signal_output.save_to(output_filename)

    return signal_output


def tune_manual(filename, output_filename, segments, attack=0.0, strength=1.0, minimum_vib_frames=8, minimum_gliss_frames=8, debug=False):
    signal = Signal.from_file(filename)

    frames = Frames(round(signal.sr/10), 0.8)
    frames.frame_signal(signal)

    pitches, energies = sweep_pitch_algorithm(frames)
    energy_range = max(energies.fx) - min(energies.fx)
    energy_threshold = (energy_range * 0.1) + min(energies.fx)

    source_markers = get_pitch_markers(frames, pitches)
    source_voiced_idx = np.where(energies[frames.get_frame_index(source_markers.markers)] > energy_threshold)
    analysis_markers = PitchMarkers(source_markers.markers[source_voiced_idx], source_markers.frequencies[source_voiced_idx])

    target_pitches = segments.get_target_pitches(len(pitches))
    target_markers = get_pitch_markers(frames, target_pitches)
    target_voiced = energies[frames.get_frame_index(target_markers.markers)] > energy_threshold

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(pitches.fx, label="Original")
        plt.plot(target_pitches.fx, color="blueviolet", label="Target")
        plt.title("Original and Target Frequencies")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Frame")
        plt.legend()
        plt.show()

    signal_output = td_psola(signal, analysis_markers, target_markers, target_voiced)

    if output_filename:
        signal_output.save_to(output_filename)

    return signal_output, pitches, target_pitches


def harmonize(filename, output_filename, tracks, chorus=False, debug=False):
    if chorus and False:
        chorus_tracks = []
        for segment in tracks:
            chorus_track = Segments.copy(segment)
            chorus_track.transpose(-0.15)
            chorus_tracks.append(chorus_track)
            chorus_track = Segments.copy(segment)
            chorus_track.transpose(0.15)
            chorus_tracks.append(chorus_track)
        tracks.extend(chorus_tracks)

    signals = []
    original_pitches, target_pitches = None, []
    for segments in tracks:
        signal_output, o_p, t_p = tune_manual(filename, None, segments)

        if chorus:
            sample_offset = np.random.randint(0, signal_output.sr / 20)
            signal_output.y = np.roll(signal_output.y, sample_offset)
            signal_output.y[:sample_offset] = 0.0

        signals.append(signal_output.y)
        original_pitches = o_p
        target_pitches.append(t_p)
    signal_harmonized = Signal(np.mean(signals, axis=0), signal_output.sr)

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(original_pitches.fx, label="Original")
        for t_p in target_pitches:
            plt.plot(t_p.fx)
        plt.title("Original and Target Frequencies")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Frame")
        plt.legend()
        plt.show()

    if output_filename:
        signal_harmonized.save_to(output_filename)

    return signal_harmonized
