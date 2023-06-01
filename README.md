# DoReMi 🎹
A minimalist implementation of a vocal autotuner in Python, with no external dependencies except for Numpy.

### Quick Start

```python
import doremi
doremi.tune("examples/original.wav", "examples/Dpentatonic.wav", key="D", scale="pentatonic")
```

That's all there is to it. All twelve notes in the Western chromatic scale are supported, and can be referenced through any valid enharmonic notation excluding double sharps and double flats (e.g. "F" is the same as "E#"). A frequency in hertz can also be passed in place of the key letter for more precise tuning control:

```python
doremi.tune("input.wav", "tuned.wav", 442, "major")
```

Valid arguments for the scale are **major**, **minor**, **chromatic**, and **pentatonic**.

To produce a diagram showing the original and output pitches, set the 'debug' argument to True:

```python
doremi.tune("examples/original.wav", "examples/Dpentatonic.wav", "D", "pentatonic", debug=True)
```

Examples of tuned samples can be found in /examples.