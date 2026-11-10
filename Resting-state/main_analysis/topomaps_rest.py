"""
Topographic maps of Cohen's d between WSN and SN for the node level metrics.

Reads rest_all_metrics.pkl, written by final_rest.py. Mann Whitney U per
channel, FDR corrected across the 16 channels within each metric and band.
Global metrics get one FDR across all metric and band pairs.

Cohen's d is WSN minus SN, so negative is higher in SN.

Saves topomap_local_metrics.svg
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import mne
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

channel_labels = [
    "Fp1","Fp2","F3","F4","Fz","FC1","FC2","Cz",
    "P1","P2","C1","C2","CP3","CP4","O1","O2"
]

# Node level, one value per channel
local_metrics = [
    "eigenvector_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "clustering_coefficient"
]

# One value per subject
global_metrics = ["efficiency", "density", "modularity", "transitivity"]

bands = ["delta", "theta", "alpha", "beta"]

alpha = 0.05
d_limit = 1.7  # colour scale limit


def channel_means(subj_id, band, metric, all_data):
    # Average each channel over the epochs
    metric_list = all_data[subj_id][band][metric]

    per_epoch = []
    for ep_dict in metric_list:
        if isinstance(ep_dict, dict):
            arr = np.full(len(channel_labels), np.nan)
            for ch_idx, val in ep_dict.items():
                if ch_idx < len(channel_labels):
                    arr[ch_idx] = val
            per_epoch.append(arr)

    if len(per_epoch) == 0:
        return np.full(len(channel_labels), np.nan)
    return np.nanmean(np.array(per_epoch), axis=0)


def global_mean(subj_id, band, metric, all_data):
    vals = [v for v in all_data[subj_id][band][metric] if v is not None]
    return np.nanmean(vals) if vals else np.nan


def cohen_d(sn_vals, wsn_vals):
    # WSN minus SN over the pooled SD
    sn_vals, wsn_vals = np.asarray(sn_vals), np.asarray(wsn_vals)
    if len(sn_vals) < 2 or len(wsn_vals) < 2:
        return np.nan

    n1, n2 = len(sn_vals), len(wsn_vals)
    v1, v2 = np.var(sn_vals, ddof=1), np.var(wsn_vals, ddof=1)
    pooled = ((n1 - 1)*v1 + (n2 - 1)*v2) / (n1 + n2 - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(wsn_vals) - np.mean(sn_vals)) / np.sqrt(pooled)


def electrode_positions():
    # Positions from the standard montage, z dropped
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()['ch_pos']

    pos = np.full((len(channel_labels), 2), np.nan)
    for i, ch in enumerate(channel_labels):
        if ch in ch_pos:
            pos[i, 0], pos[i, 1] = ch_pos[ch][0], ch_pos[ch][1]
    return pos


if __name__ == "__main__":

    with open("rest_all_metrics.pkl", "rb") as f:
        all_metrics = pickle.load(f)

    subjects = sorted(all_metrics.keys())
    sn_ids  = [s for s in subjects if s.startswith("SN_")]
    wsn_ids = [s for s in subjects if s.startswith("WSN_")]
    print(f"{len(sn_ids)} SN and {len(wsn_ids)} WSN subjects")

    pos2d = electrode_positions()

    # Effect size and raw p per channel
    local_results = {}
    for metric in local_metrics:
        for band in bands:
            sn_mat  = np.array([channel_means(s, band, metric, all_metrics) for s in sn_ids])
            wsn_mat = np.array([channel_means(s, band, metric, all_metrics) for s in wsn_ids])

            d_map = np.full(len(channel_labels), np.nan)
            p_map = np.full(len(channel_labels), np.nan)

            for ch in range(len(channel_labels)):
                g1 = sn_mat[:, ch][~np.isnan(sn_mat[:, ch])]
                g2 = wsn_mat[:, ch][~np.isnan(wsn_mat[:, ch])]
                if len(g1) > 1 and len(g2) > 1:
                    d_map[ch] = cohen_d(g1, g2)
                    _, p_map[ch] = mannwhitneyu(g1, g2, alternative='two-sided')

            local_results[(metric, band)] = (d_map, p_map)

    # Metrics down the rows, bands across the columns
    fig, axes = plt.subplots(
        nrows=len(local_metrics), ncols=len(bands),
        figsize=(4*len(bands), 4*len(local_metrics)), facecolor='white'
    )

    im_for_cbar = None

    for row, metric in enumerate(local_metrics):
        for col, band in enumerate(bands):
            ax = axes[row, col]
            d_map, p_map = local_results[(metric, band)]

            valid = ~np.isnan(p_map)
            reject_map = np.full(len(channel_labels), False)
            reject, _ = fdrcorrection(p_map[valid], alpha=alpha)
            reject_map[valid] = reject

            im, _ = mne.viz.plot_topomap(
                data=d_map, pos=pos2d, axes=ax, cmap='RdBu_r',
                outlines='head', sensors=True, contours=0, show=False
            )
            im.set_clim(-d_limit, d_limit)
            im_for_cbar = im

            for ch, ch_name in enumerate(channel_labels):
                xx, yy = pos2d[ch]
                if np.isnan(xx):
                    continue
                ax.text(xx, yy + 0.01, ch_name, color='black', fontsize=12,
                        ha='center', va='center')
                if reject_map[ch]:
                    ax.text(xx, yy - 0.01, '*', color='white',
                            ha='center', va='bottom', fontsize=12)

            if row == 0:
                ax.set_title(f"{band} band", fontweight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel(metric, fontweight='bold', fontsize=10)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Cohen's d (WSN - SN)", fontsize=10)

    plt.savefig("topomap_local_metrics.svg", format="svg", bbox_inches="tight")
    plt.close()
    print("saved topomap_local_metrics.svg")

    # Per channel results
    for metric in local_metrics:
        for band in bands:
            d_map, p_map = local_results[(metric, band)]
            valid = ~np.isnan(p_map)
            reject, p_fdr = fdrcorrection(p_map[valid], alpha=alpha)

            reject_map = np.full(len(channel_labels), False)
            fdr_map = np.full(len(channel_labels), np.nan)
            reject_map[valid] = reject
            fdr_map[valid] = p_fdr

            print(f"\n{metric}, {band} band")
            for ch, ch_name in enumerate(channel_labels):
                if np.isnan(p_map[ch]):
                    continue
                flag = " *" if reject_map[ch] else ""
                print(f"  {ch_name:<4s} d {d_map[ch]:6.3f}   p {p_map[ch]:.4g}"
                      f"   FDR p {fdr_map[ch]:.4g}{flag}")

    # Global metrics, one FDR over all of them
    keys = [(m, b) for m in global_metrics for b in bands]
    d_vals, p_vals = [], []

    for metric, band in keys:
        sn_vals  = np.array([global_mean(s, band, metric, all_metrics) for s in sn_ids])
        wsn_vals = np.array([global_mean(s, band, metric, all_metrics) for s in wsn_ids])
        sn_vals  = sn_vals[~np.isnan(sn_vals)]
        wsn_vals = wsn_vals[~np.isnan(wsn_vals)]

        if len(sn_vals) < 2 or len(wsn_vals) < 2:
            d_vals.append(np.nan)
            p_vals.append(np.nan)
        else:
            d_vals.append(cohen_d(sn_vals, wsn_vals))
            p_vals.append(mannwhitneyu(sn_vals, wsn_vals, alternative='two-sided')[1])

    reject_g, p_fdr_g = fdrcorrection(p_vals, alpha=alpha)

    print("\nGlobal metrics")
    for i, (metric, band) in enumerate(keys):
        flag = " *" if reject_g[i] else ""
        print(f"  {metric:<14s} {band:<6s} d {d_vals[i]:6.3f}   p {p_vals[i]:.4g}"
              f"   FDR p {p_fdr_g[i]:.4g}{flag}")
