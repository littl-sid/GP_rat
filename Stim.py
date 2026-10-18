import numpy as np
import sounddevice as sd
import time
import os
from IPython import embed

os.makedirs("GP_rat/stim", exist_ok=True)
RAT = 5
POS = 4

rng = np.random.default_rng()

sample_rate = 44100  # Hz
trials = 5           # trials (repetitions of sounds)
duration = 2.0       # duration of stimulation
delay = 5.0          # delay duration

frequencies = np.array([1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 50, 90, 170, 330, 650, 800])
#frequencies = np.array([500, 600])
LF = len(frequencies)

T = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# c_T = np.linspace(0, delay, int(sample_rate * delay), False)
# c_tone_sin = 0.5 * np.sin(2 * np.pi * 0 * c_T)
c_tone_zeros = np.zeros(int(sample_rate * delay))

# print(np.array_equal(c_tone_sin, c_tone_zeros)) -> True

freq_sequence = np.empty(LF * trials)
start_time = time.perf_counter()
for t in range(trials):
    rand_frequencies = rng.permutation(frequencies)
    for i, f in enumerate(rand_frequencies):
        # freq_sequence[i + t * LF - 1] = f index fehler
        freq_sequence[i + t * LF] = f

        print(f)
        tone = 0.5 * np.sin(2 * np.pi * f * T)

        # 5 s 0 Hz & 2 s stim
        stimulus = np.concatenate((c_tone_zeros, tone))

        # replay both
        sd.play(stimulus, sample_rate)
        sd.wait()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f'Elapsed time: {elapsed_time} seconds'
      f' Ideal time: {(delay + duration) * len(frequencies) * trials} seconds')

# Gesamte Sequenz inklusive Stille
full_stim = np.empty(len(freq_sequence) * 2, dtype=int)
full_stim[0::2] = 0
full_stim[1::2] = freq_sequence

np.save(f"GP_rat/stim/r{RAT}_p{POS}_stim_full.npy", full_stim)
np.save(f'GP_rat/stim/r{RAT}_p{POS}_stim.npy', freq_sequence)
np.save(f'GP_rat/stim/r{RAT}_p{POS}_dur.npy', elapsed_time)
