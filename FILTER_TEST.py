import doremi


signal = doremi.Signal.from_file("examples/manual_tune/harmonize5.wav")

reverb_signal = doremi.reverb(signal, wet_mix=0.5)

reverb_signal.save_to("examples/reverb_chorus3.wav")