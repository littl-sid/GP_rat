import numpy as np
import matplotlib.pyplot as plt
from nex5file.reader import Reader
from IPython import embed


def get_data(path, rat, pos):
    """
    get base data to work with

    INPUT
    path:   string, data path
    rat:    int,    number of the rat 
    pos:    int,    stimulated position

    OUTPUT
    stim:       array,  stimulus only 2s frequencies
    stim_full:  array,  full stimulus with 0 Hz breaks

    neurons:    array,  names of the different neurons
    spikes:     array,  timestamps of the spikes of the neurons

    cont_vals:  array,  continous values -> not sure of what
    sr:         float,  sampling rate
    t:          array,  time
    """

    # count fragments -> should be 1

    # Stimulus data
    stim_npy = np.load(f"{path}/stim/r{rat}_p{pos}_stim.npy")
    stim_full = np.load(f"{path}/stim/r{rat}_p{pos}_stim_full.npy")

    # Dataset
    nex = Reader().ReadNexFile(f"{path}/data/r{rat}_p{pos}.nex")

    if len(nex["ad64"].FragmentCounts()) == 1:
        print("1 Fragment :)")
    else: print("error in Dataset")

    # Values
    neuronnames = nex.NeuronNames() # singled out neurons
    spikes = {n: nex[n].Timestamps() for n in nex.NeuronNames()}

    stim = nex["ad64"] # continous signal
    stim_vals  = stim.ContinuousValues()
    stim_sr = stim.SamplingRate()
    stim_t  = stim.FragmentTimestamps()[0] + np.arange(len(stim_vals)) / stim_sr

    data = {
        "stim_npy": stim_npy,
        "stim_npy_full": stim_full,
        "neurons": neuronnames,
        "spikes": spikes,
        "stim_vals": stim_vals,
        "stim_sr": stim_sr,
        "stim_t": stim_t
    }
    stim_sr = nex["ad64"].SamplingRate()
    print("Samplingrate:", stim_sr, "Hz")

    return data


if __name__ == "__main__":
    path = "GP_rat"
    data = get_data(path, 1, 1)