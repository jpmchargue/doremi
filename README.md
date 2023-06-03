# DoReMi 🎹
A minimalist implementation of a vocal autotuner in Python, with no external dependencies except for Numpy.

## Quick Start

Import *doremi* from the command line using PyPI:

```
pip install doremi
```

Then tune away:

```python
import doremi
doremi.tune("examples/original.wav", "examples/Dmajor.wav", key="D", scale="major")
```

That's all there is to it. All twelve notes in the Western chromatic scale are supported, and can be referenced through any valid enharmonic notation excluding double sharps and double flats (e.g. "F" is the same as "E#"). A frequency in hertz can also be passed in place of the key letter for more precise tuning control:

```python
doremi.tune("input.wav", "tuned.wav", 442, "major")
```

Valid arguments for the scale are **major**, **minor**, **chromatic**, and **pentatonic**.

Additional arguments for the *attack* of the tuning effect (i.e. the number of seconds before the tuning takes full effect) and the *strength* of the tuning effect (i.e. how much to apply the tuning, on a scale from 0 to 1) can also be passed in. For instance, setting *attack* to 0.1 and *strength* to 0.9 avoids the robotic 'AutoTune' effect, making the tuning more subtle:

```python
doremi.tune("examples/original.wav", "examples/Gpentatonic.wav", key="G", scale="pentatonic", attack=0.1, strength=0.9)
```

To produce a diagram showing the original and output pitches, set the 'debug' argument to True:

```python
doremi.tune("examples/original.wav", "examples/Dpentatonic.wav", "D", "pentatonic", debug=True)
```

Examples of tuned samples can be found in /examples.