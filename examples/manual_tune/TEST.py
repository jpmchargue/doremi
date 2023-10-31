import doremi 


s1 = doremi.Segments([554, 659, 554, 659, 554, 494], [0, 120, 160, 210, 222, 295])
s2 = doremi.Segments([330, 415, 440], [0, 120, 210])
s3 = doremi.Segments([220, 277, 294, 330], [0, 120, 210, 295])

s1.transpose(-4)
s2.transpose(-4)
s3.transpose(-4)

doremi.harmonize("examples/original.wav", "testing/harmonize2.wav", [s1, s2, s3], debug=True)