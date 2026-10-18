import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kruskal
# from pathlib import Path
# import csv
# from matplotlib.patches import Patch
# from scipy.ndimage import binary_closing
from IPython import embed

from preprocess import get_data
import functions as fun

plt.rcParams.update({
    'font.size': 12,          # Allgemeine Schriftgröße
    'axes.titlesize': 16,     # Titel der Achsen
    'axes.labelsize': 14,     # X- und Y-Achsenbeschriftung
    'xtick.labelsize': 12,    # X-Achsen-Tick-Labels
    'ytick.labelsize': 12,    # Y-Achsen-Tick-Labels
    'legend.fontsize': 12,    # Legendentext
    'legend.title_fontsize': 13,  # Legendentitel
    'figure.titlesize': 18,   # Übergeordneter Figure-Titel (suptitle)
})

# COLORS
c_blue = "#5a93a6"    
c_red = "#60212e"       
c_beige = "#d8d1bd"  

# PATHS
stim_path = "GP_rat/stim"
res_path = "GP_rat/results"
data_path = "GP_rat/data"
path = "GP_rat"
RAT_LIST = [[2,1], [2,2], [3,1], [4,1], [4,2], [5,1], [5,2], [5,3]]


for i in RAT_LIST:
        
    rat = i[0]
    pos = i[1]


    data = get_data(f"{path}", rat, pos)

    """
    data:
    "stim_npy": stim,
    "stim_npy_full": stim_full,
    "duration": dur,
    "neurons": neuronnames,
    "spikes": spikes,
    "stim_vals": stim_vals,
    "stim_sr": stim_sr,
    "stim_t": stim_t
    """


    # ==============================
    # FREQUENCY PROCESSING
    # ==============================
    # Get freqs a& check with original from stim script
    fun.get_freqs(data, rat, pos)
    onsets, freqs = np.loadtxt(f"{stim_path}/r{rat}_p{pos}_analysed_stim.csv", delimiter=",", skiprows=1, unpack=True)

    # ==============================
    # # PLOT TIME COURSE OVERALL
    # ==============================
    # stimulus from nex file
    stim_t = data["stim_t"]
    stim_sr = data["stim_sr"]
    stim_vals = data["stim_vals"]

    # spikes
    spikes = data["spikes"][data["neurons"][0]]

    # Plot over time of raw data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), layout="constrained")
    ax1.plot(stim_t, stim_vals, color="k")
    ax1.set_ylabel("Stimulus [mV]")
    ax1.set_title(f"Rat: {rat}, Position: {pos}")

    ax2.eventplot(spikes, lineoffsets=0, linelengths=0.8, color="k")
    ax2.set_yticks([])   
    ax2.set_ylabel("neuron")
    ax2.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(f"{res_path}/fig_timecourse_r{rat}_p{pos}.png", bbox_inches="tight", dpi=300)


    # ==============================
    # TUNING CURVES
    # ==============================
    tone_dur = 2.0 # s, duration tone widow
    base_dur = 2.0 # s, duration baseline window during silence
    base_gap = 1.5 # s, gap of base window before tone

    possible_freqs = np.unique(freqs)

    fig, axes = plt.subplots(1, len(data["neurons"]), figsize=(12, 8), squeeze=False, layout="constrained")

    for ax, neuron in zip(axes[0], data["neurons"]):
        neuron_spikes = data["spikes"][neuron]

        tone_rates = []
        base_rates = []
        for onset in onsets:
            # spikes during tone n
            n_tone = np.sum((neuron_spikes >= onset) & (neuron_spikes < onset + tone_dur))
            # spikes during silence
            base_start = onset - base_gap - base_dur
            n_base = np.sum((neuron_spikes >= base_start) & (neuron_spikes < base_start + base_dur))
            tone_rates.append(n_tone / tone_dur)
            base_rates.append(n_base / base_dur)
        tone_rates = np.array(tone_rates)
        base_rate = np.mean(base_rates)

        # per freq: mean & SEM
        mean_rate = np.array([tone_rates[freqs == f].mean() for f in possible_freqs])
        sem_rate = np.array([tone_rates[freqs == f].std(ddof=1) / np.sqrt(np.sum(freqs == f))
                            for f in possible_freqs])

        try:
            p_tuning = kruskal(*[tone_rates[freqs == f] for f in possible_freqs]).pvalue
        except ValueError: # all rates identical (e.g. no spikes at all)
            p_tuning = np.nan

        # Plot
        ax.errorbar(possible_freqs, mean_rate, yerr=sem_rate, marker="o", color=c_red, label=r"mean $\pm$ SEM")
        ax.axhline(base_rate, color="k", linestyle="--", label="baseline")

        ax.set_xscale("log")
        ax.set_xticks(possible_freqs)
        ax.set_xticklabels([f"{f:g}" for f in possible_freqs])
        ax.tick_params(axis="x", rotation=90)

        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("rate [Hz]")
        ax.set_title(f"Rat: {rat}, Position: {pos}, p = {p_tuning:.3f}")

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    fig.tight_layout()
    fig.savefig(f"{res_path}/fig_tuning_r{rat}_p{pos}.png", bbox_inches="tight", dpi=300)


    # ==============================
    # TIME COURSE OF STIM
    # ==============================
    cmap = LinearSegmentedColormap.from_list("freq", [c_blue, c_red])   # no black middle

    bin_width = 0.5 # s
    bin_edges = np.arange(-(base_gap + base_dur), 3 + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2


    fig, axes = plt.subplots(1, len(data["neurons"]), figsize=(12, 8), squeeze=False, layout="constrained")

    selected = {}

    for ax, neuron in zip(axes[0], data["neurons"]):
        neuron_spikes = data["spikes"][neuron]
        rates = np.array([np.histogram(neuron_spikes - o, bin_edges)[0] / bin_width for o in onsets])

        selected[neuron] = possible_freqs[mean_rate > 1.5 * base_rate + 1]   # at least 1 Hz above 1.5 x baseline
        selected[neuron] = possible_freqs

        for f in selected[neuron]:
            color = cmap(np.log(f) / np.log(possible_freqs.max()))
            ax.plot(bin_centers, rates[freqs == f].mean(axis=0), color=color, lw=1, label=f"{f:g} Hz")

        if len(selected[neuron]):                                            # mean over all selected frequencies
            is_selected = np.isin(freqs, selected[neuron])
            ax.plot(bin_centers, rates[is_selected].mean(axis=0), color="k", lw=3, label="mean selected")

        ax.axhline(base_rate, color="k", ls="--", lw=0.8)                    # baseline
        ax.axvspan(0, tone_dur, color=c_beige, zorder=0) # tone
        ax.set_xlabel("time from tone onset [s]")
        ax.set_ylabel("rate [Hz]")
        ax.set_title(f"Rat: {rat}, Position: {pos}")

        if len(selected[neuron]):
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="frequency")

    fig.tight_layout()
    fig.savefig(f"{res_path}/fig_stimreaction_r{rat}_p{pos}.png", bbox_inches="tight", dpi=300)


    # ==============================
    # PHASE LOCKING
    # ==============================
    spike_start, spike_len = 0.2, 1.8 # s; window starts 0.2 s after onset (skips the onset click)
    sig_alpha = 0.05 / len(possible_freqs) # 16 frequencies are tested

    def stim_phase(onset, f):
        """phase of the sine in the ad64 signal, fitted over the clean part of the tone"""
        m = (stim_t >= onset + 0.2) & (stim_t < onset + 1.8)
        z = np.sum((stim_vals[m] - stim_vals[m].mean()) * np.exp(-2j * np.pi * f * stim_t[m]))
        return np.angle(z)


    fig, axes = plt.subplots(2, len(data["neurons"]), figsize=(12, 8), squeeze=False, layout="constrained")

    for col, neuron in enumerate(data["neurons"]):
        ax_phase, ax_vs = axes[0, col], axes[1, col]
        neuron_spikes = data["spikes"][neuron]
        vector_strength, p_value = [], []

        for k, f in enumerate(possible_freqs):
            phases = []
            for onset in onsets[freqs == f]:
                spike_end = spike_start + np.floor(f * spike_len) / f # whole cycles only
                s = neuron_spikes[(neuron_spikes >= onset + spike_start) & (neuron_spikes < onset + spike_end)]
                phases.append((2 * np.pi * f * s + stim_phase(onset, f)) % (2 * np.pi))   # phase of the stimulus at each spike
            phases = np.concatenate(phases)

            ax_phase.plot(phases, np.full(len(phases), k), "|", color=c_red, alpha=0.5, ms=6)

            n = len(phases)
            R = np.abs(np.mean(np.exp(1j * phases))) if n > 0 else np.nan
            vector_strength.append(R)
            p_value.append(np.exp(-n * R**2) if n > 0 else np.nan)

        vector_strength, p_value = np.array(vector_strength), np.array(p_value)

        ax_phase.set_yticks(range(len(possible_freqs)))
        ax_phase.set_yticklabels([f"{f:g}" for f in possible_freqs])
        ax_phase.set_xlim(0, 2 * np.pi)
        ax_phase.set_xticks([0, np.pi, 2 * np.pi]); ax_phase.set_xticklabels(["0", "π", "2π"])
        ax_phase.set_xlabel("stimulus phase [rad]"); ax_phase.set_ylabel("frequency [Hz]")
        ax_phase.set_title(f"Rat: {rat}, Position: {pos}")

        signif = p_value < sig_alpha
        ax_vs.plot(possible_freqs, vector_strength, "-", color="k", lw=0.8)
        ax_vs.plot(possible_freqs[signif], vector_strength[signif], "o", color=c_red, label=f"p < {sig_alpha}")
        ax_vs.plot(possible_freqs[~signif], vector_strength[~signif], "o", mfc="none", color="k", label="n.s.")
        ax_vs.set_xscale("log")
        ax_vs.set_xticks(possible_freqs); ax_vs.set_xticklabels([f"{f:g}" for f in possible_freqs])
        ax_vs.minorticks_off(); ax_vs.tick_params(axis="x", rotation=90)
        ax_vs.set_xlabel("frequency [Hz]"); ax_vs.set_ylabel("vector strength"); ax_vs.set_ylim(0, 1)

        ax_vs.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    fig.tight_layout()
    fig.savefig(f"{res_path}/fig_phaselocking_r{rat}_p{pos}.png", bbox_inches="tight", dpi=300)


    # ==============================
    # AUTO & CROSS CORRELATIONS
    # ==============================
    # AUTO: classic autocorrelogram of the spike times (ms)
    ac_bin = 0.001 # s, 1 ms bins
    ac_lag = 0.05 # s, +-50 ms
    ac_edges = np.arange(-ac_lag, ac_lag + ac_bin, ac_bin)
    ac_centers = (ac_edges[:-1] + ac_bin / 2) * 1000 # ms

    # CROSS: spike rate x stimulus on/off (s)
    corr_bin = 0.1 # s, bin width of the spike rate
    corr_lag = 10.0 # s, lags shown (the tone repeats every ~7 s)
    n_shuffles = 1000

    corr_edges = np.arange(stim_t[0], stim_t[-1], corr_bin)
    corr_centers = corr_edges[:-1] + corr_bin / 2
    n_lags = int(corr_lag / corr_bin)
    corr_lags = np.arange(-n_lags, n_lags + 1) * corr_bin # s
    rng = np.random.default_rng(0)

    # stimulus: 1 while a tone is playing, 0 otherwise (from the onsets, so weak tones count too)
    stim_on = np.zeros(len(corr_centers))
    for onset in onsets:
        stim_on[(corr_centers >= onset) & (corr_centers < onset + tone_dur)] = 1

    def zscore(x):
        return (x - x.mean()) / x.std()

    def cross_corr(x, y):
        """correlation r between x(t) and y(t + lag) for lags -n_lags ... +n_lags (x, y z-scored)"""
        c = np.fft.irfft(np.conj(np.fft.rfft(x)) * np.fft.rfft(y), len(x)) / len(x)
        return np.concatenate([c[-n_lags:], c[:n_lags + 1]])

    stim_z = zscore(stim_on)

    fig, axes = plt.subplots(2, len(data["neurons"]), figsize=(12, 8), squeeze=False,layout="constrained")

    for col, neuron in enumerate(data["neurons"]):
        ax_auto, ax_cross = axes[0, col], axes[1, col]
        neuron_spikes = data["spikes"][neuron]

        # --- AUTO: neuron x itself ---
        # time difference from every spike to all spikes within +-50 ms
        diffs = np.concatenate([neuron_spikes[np.searchsorted(neuron_spikes, t - ac_lag):np.searchsorted(neuron_spikes, t + ac_lag)] - t
                                for t in neuron_spikes])
        diffs = diffs[diffs != 0] # remove each spike paired with itself
        ac = np.histogram(diffs, ac_edges)[0] / (len(neuron_spikes) * ac_bin) # rate [Hz] at each lag, given a spike at lag 0
        mean_rate_all = len(neuron_spikes) / (stim_t[-1] - stim_t[0])

        ax_auto.bar(ac_centers, ac, width=ac_bin * 1000, color=c_red)
        ax_auto.axhline(mean_rate_all, color="k", ls="--", lw=0.8, label="mean rate")
        ax_auto.set_xlabel("lag [ms]"); ax_auto.set_ylabel("rate [Hz]")
        ax_auto.set_title(f"Rat: {rat}, Position {pos} - autocorrelogram")
        
        ax_auto.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        print(f"{neuron}: {len(neuron_spikes)} spikes, mean rate {mean_rate_all:.1f} Hz, "
              f"ISI < 2 ms: {100 * np.mean(np.diff(neuron_spikes) < 0.002):.2f} %")

        # --- CROSS: neuron x stimulus ---
        rate_z = zscore(np.histogram(neuron_spikes, corr_edges)[0] / corr_bin)
        r = cross_corr(stim_z, rate_z)
        # chance: shift the spike rate by a random time (keeps its structure, breaks the link to the stimulus)
        null_cross = np.array([np.abs(cross_corr(stim_z, np.roll(rate_z, rng.integers(int(20 / corr_bin), len(rate_z) - int(20 / corr_bin))))).max()
                               for _ in range(n_shuffles)])
        thr_cross = np.percentile(null_cross, 95)
        ax_cross.axhspan(-thr_cross, thr_cross, color=c_beige, zorder=0, label="chance (95 %)")
        ax_cross.plot(corr_lags, r, color=c_red)
        ax_cross.axvline(0, color="k", lw=0.5); ax_cross.axhline(0, color="k", lw=0.5)
        ax_cross.set_xlabel("lag [s]  (>0: neuron follows stimulus)"); ax_cross.set_ylabel("cross-correlation r")
        ax_cross.set_title(f"Rat: {rat}, Position {pos} - neuron x stimulus")
        
        ax_cross.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        peak = np.argmax(np.abs(r))
        print(f"{neuron} cross: peak r = {r[peak]:.3f} at lag {corr_lags[peak]:+.1f} s, "
              f"p = {np.mean(null_cross >= abs(r[peak])):.3f} (chance level |r| < {thr_cross:.3f})")

        

    fig.tight_layout()
    fig.savefig(f"{res_path}/fig_correlations_r{rat}_p{pos}.png", bbox_inches="tight", dpi=300)



    # ==============================
    # COORDINATES
    # ==============================
    