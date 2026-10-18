import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from matplotlib.patches import Patch
from scipy.ndimage import binary_closing
from IPython import embed

from preprocess import get_data

stim_path = "GP_rat/stim"
res_path = "GP_rat/results"
data_path = "GP_rat/data"


def get_freqs(data, rat, pos):
    stim_t = data["stim_t"]
    stim_sr = data["stim_sr"]
    stim_vals = data["stim_vals"]

    # CHECK STIMULUS
    # Set zero line to 0 & estimate base noise
    signal = stim_vals - np.median(stim_vals)
    noise_level = np.median(np.abs(signal)) / 0.6745 # 75% quantile

    # Search tone -> tone = signal significant over noise -> true or false
    is_loud = np.abs(signal) > 20 * noise_level
    is_loud = binary_closing(is_loud, np.ones(int(0.6 * stim_sr)))   # Nulldurchgänge überbrücken

    change = np.diff(is_loud.astype(int))                  # +1 = tone starts, -1 = tone ends
    burst_start = stim_t[np.flatnonzero(change == 1) + 1] 
    burst_end   = stim_t[np.flatnonzero(change == -1) + 1]
    # start and end has to be the same
    assert len(burst_start) == len(burst_end), "tone cut off at start or end of recording"

    is_tone = np.isclose(burst_end - burst_start, 2.0, atol=0.5)  # keep only bursts of ~2 s (1.5 - 2.5 s)
    burst_start = burst_start[is_tone]

    # Grid for tones (since we have even spacing)
    tone_nr = np.round((burst_start - burst_start[0]) / np.median(np.diff(burst_start)))
    period, first_onset = np.polyfit(tone_nr, burst_start, 1)

    # First found tone counts as trial 0 (for now)
    residuals = burst_start - (first_onset + tone_nr * period)
    print(f"period {period:.3f} s, max deviation {np.abs(residuals).max():.3f} s (should be < 0.05)")

    expected_freqs = data["stim_npy"]            # freq per trial from Stim.py
    possible_freqs = np.unique(expected_freqs)
    onsets = first_onset + np.arange(len(expected_freqs)) * period

    # Spectrum settings
    freq_step = 0.05                             # Hz between spectrum points
    n_samples = int(stim_sr / freq_step)
    freq_axis = np.fft.rfftfreq(n_samples, 1 / stim_sr)
    window_samples = int(1.6 * stim_sr)          # 1.6 s piece, skips the clicks at start/end of the tone

    def get_spectrum(piece):
        piece = piece - piece.mean()
        piece = piece * np.hanning(len(piece))
        return np.abs(np.fft.rfft(piece, n_samples)) ** 2

    def measure_freqs(onsets):
        measured = []
        for tone in onsets:
            tone_start = int((tone + 0.2 - stim_t[0]) * stim_sr)
            silence_start = int((tone - 0.5 - 1.6 - stim_t[0]) * stim_sr)   # silence before the tone
            if silence_start < 0:
                silence_start = int((tone + 2.5 - stim_t[0]) * stim_sr)     # fallback: silence after
            if min(tone_start, silence_start) < 0 or max(tone_start, silence_start) + window_samples > len(signal):
                measured.append(np.nan)                                     # window outside the recording
                continue
            tone_spectrum = get_spectrum(signal[tone_start : tone_start + window_samples])
            silence_spectrum = get_spectrum(signal[silence_start : silence_start + window_samples])
            ratios = []
            for freq in possible_freqs:
                near_freq = (freq_axis > freq - 0.7) & (freq_axis < freq + 0.7)
                ratios.append(tone_spectrum[near_freq].max() / silence_spectrum[near_freq].max())
            measured.append(possible_freqs[np.argmax(ratios)])
        return np.array(measured)

    # Try the grid shifted by whole trials, keep the best
    match_rates = {}
    for shift in range(-3, 4):
        measured_freqs = measure_freqs(onsets + shift * period)
        valid = ~np.isnan(measured_freqs)
        match_rates[shift] = np.mean(np.isclose(measured_freqs[valid], expected_freqs[valid], atol=1))
        # print(f"shift {shift:+d} trials: {100 * match_rates[shift]:.0f} % ({valid.sum()} valid trials)")

    best_shift = max(match_rates, key=match_rates.get)
    onsets = onsets + best_shift * period
    assert onsets[-1] + 2.0 <= stim_t[-1], "grid runs past end of recording"

    measured_freqs = measure_freqs(onsets)
    match = np.isclose(measured_freqs, expected_freqs, atol=1)
    print(f"best shift {best_shift:+d} -> stimulus match: {100 * np.nanmean(match):.0f} %")

    np.savetxt(f"{stim_path}/r{rat}_p{pos}_analysed_stim.csv",
            np.column_stack([onsets, measured_freqs]),
            delimiter=",", header="onset_s,freq_Hz", comments="")