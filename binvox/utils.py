
def compress_flat_voxels(voxels_flat):
    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    counter = 0
    compressed = []

    for c in voxels_flat:
        if c == state:
            counter += 1
            # if counter hits max, dump
            if counter == 255:
                compressed += [int(state), counter]
                counter = 0
        else:
            # if switch state, dump
            compressed += [int(state), counter]
            state = c
            counter = 1
    # flush out remainders
    if counter > 0:
        compressed += [int(state), counter]

    return compressed
