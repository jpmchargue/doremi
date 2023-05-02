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