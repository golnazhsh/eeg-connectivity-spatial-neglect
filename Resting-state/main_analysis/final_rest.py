"""
Resting state classification, SN against WSN, from coherence graph metrics.

Loads the coherence mat files, drops outlier epochs per subject, then runs
leave one participant out cross validation. Otsu thresholds and RFE are fit
inside each fold on the training subjects only.

Saves lopo_results_rest_nested.pkl and rest_all_metrics.pkl.
"""

import os
import numpy as np
import scipy.io
import pickle
import pandas as pd
import networkx as nx
import inspect

from skimage.filters import threshold_otsu
from networkx.exception import PowerIterationFailedConvergence

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE

# Folder holding the coherence mat files
data_path = '../connectivity'

# One entry per session, in the same order as the cell arrays in the mat files
SN_subject_ids = [
    "SN_108","SN_108","SN_109","SN_109",
    "SN_110","SN_110","SN_111","SN_111","SN_112","SN_112",
    "SN_112","SN_113","SN_113","SN_114","SN_114",
    "SN_114","SN_115","SN_115","SN_116","SN_116",
    "SN_116","SN_117","SN_117"
]

WSN_subject_ids = [
    "WSN_115","WSN_115","WSN_116",
    "WSN_116","WSN_117","WSN_117","WSN_118","WSN_118",
    "WSN_118","WSN_119","WSN_119","WSN_119",
    "WSN_120"
]

rest_paths = {
    'SN': {
        'delta': os.path.join(data_path, 'SN_deltaCoh.mat'),
        'theta': os.path.join(data_path, 'SN_thetaCoh.mat'),
        'alpha': os.path.join(data_path, 'SN_alphaCoh.mat'),
        'beta':  os.path.join(data_path, 'SN_betaCoh.mat'),
    },
    'WSN': {
        'delta': os.path.join(data_path, 'WSN_deltaCoh.mat'),
        'theta': os.path.join(data_path, 'WSN_thetaCoh.mat'),
        'alpha': os.path.join(data_path, 'WSN_alphaCoh.mat'),
        'beta':  os.path.join(data_path, 'WSN_betaCoh.mat'),
    }
}

# Resting state has no gamma
bands = ["delta", "theta", "alpha", "beta"]

local_metrics = [
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "clustering_coefficient"
]

global_metrics = [
    "efficiency",
    "density",
    "modularity",
    "transitivity"
]

channel_labels = [
    "Fp1","Fp2","F3","F4","Fz","FC1","FC2","Cz",
    "P1","P2","C1","C2","CP3","CP4","O1","O2"
]

# CP3 and CP4 sit in the central regions here
channel_groups = {
    "L_Frontal":  ["Fp1","F3"],
    "R_Frontal":  ["Fp2","F4"],
    "L_Central":  ["FC1","CP3","C1"],
    "R_Central":  ["FC2","CP4","C2"],
    "L_ParOcc":   ["P1","O1"],
    "R_ParOcc":   ["P2","O2"],
    "Midline":    ["Fz","Cz"]
}

random_state = 88
n_features_to_select = 30

rf_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

other_models = {
    "SVM (RBF)":    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=random_state),
    "GradientBoost": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    "LDA":          LinearDiscriminantAnalysis(),
    "NaiveBayes":   GaussianNB()
}

SAVE_PATH = "."


def extract_list_of_arrays(mat_data):
    # One 3D array per session, channels by channels by epochs
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    data_struct = mat_data[keys[0]]
    return [elem for elem in data_struct[0]]


def remove_epoch_outliers_iqr(coh_array, subj_id, band, iqr_factor=0.75):
    # Drop epochs whose mean coherence falls outside the IQR fence, per subject
    n_ep = coh_array.shape[-1]

    ep_means = np.full(n_ep, np.nan, dtype=np.float32)
    for ep in range(n_ep):
        mat = coh_array[..., ep]
        vals = mat[np.isfinite(mat)]
        if len(vals) > 0:
            ep_means[ep] = vals.mean()

    valid_mask = ~np.isnan(ep_means)
    ep_means_valid = ep_means[valid_mask]
    keep_ep = np.where(valid_mask)[0]

    q1 = np.percentile(ep_means_valid, 25)
    q3 = np.percentile(ep_means_valid, 75)
    iqr_val = q3 - q1
    low_f  = q1 - iqr_factor * iqr_val
    high_f = q3 + iqr_factor * iqr_val

    remove_mask = (ep_means_valid < low_f) | (ep_means_valid > high_f)
    final_keep_ep = keep_ep[np.where(~remove_mask)[0]]

    return coh_array[..., final_keep_ep]


def load_and_group_by_subject(paths_dict, sn_ids, wsn_ids):
    # Read the mat files, drop outlier epochs, group the sessions by subject
    data_by_subj = {}

    for group, ids in [('SN', sn_ids), ('WSN', wsn_ids)]:
        for band, fpath in paths_dict[group].items():
            mat_data = scipy.io.loadmat(fpath)
            subject_matrices = extract_list_of_arrays(mat_data)

            for idx, subj_id in enumerate(ids):
                arr = remove_epoch_outliers_iqr(subject_matrices[idx], subj_id, band)
                data_by_subj.setdefault(subj_id, {})
                if band in data_by_subj[subj_id]:
                    arr = np.concatenate([data_by_subj[subj_id][band], arr], axis=-1)
                data_by_subj[subj_id][band] = arr

    return data_by_subj


def compute_otsu_thresholds_for_training(data_by_subj, train_subj_ids):
    # One threshold per band, from the training subjects only
    thresholds = {}
    for band in bands:
        vals = []
        for sid in train_subj_ids:
            avg_mat = np.mean(data_by_subj[sid][band], axis=-1)
            band_vals = avg_mat[avg_mat > 0].ravel()
            if band_vals.size > 0:
                vals.append(band_vals)
        thresholds[band] = threshold_otsu(np.concatenate(vals))
    return thresholds


def binarize_for_subjects(data_by_subj, subj_ids, thresholds):
    # Above threshold becomes an edge, below becomes nothing
    out = {}
    for sid in subj_ids:
        out[sid] = {band: (arr > thresholds[band]).astype(np.float32)
                    for band, arr in data_by_subj[sid].items()}
    return out


def construct_graphs_for_one_subject(subject_array):
    # One graph per epoch
    return [nx.from_numpy_array(subject_array[:, :, ep])
            for ep in range(subject_array.shape[2])]


def calculate_graph_metrics_for_one_subject(graphs):
    # Eight metrics, one value per epoch. Node level ones come back as a
    # dict keyed by channel index.
    metrics = {
        "betweenness_centrality": [],
        "closeness_centrality": [],
        "eigenvector_centrality": [],
        "clustering_coefficient": [],
        "modularity": [],
        "transitivity": [],
        "density": [],
        "efficiency": []
    }

    for G in graphs:
        metrics["betweenness_centrality"].append(nx.betweenness_centrality(G))
        metrics["closeness_centrality"].append(nx.closeness_centrality(G))

        # eigenvector does not always converge
        try:
            metrics["eigenvector_centrality"].append(nx.eigenvector_centrality(G, max_iter=2000))
        except PowerIterationFailedConvergence:
            metrics["eigenvector_centrality"].append(None)

        metrics["clustering_coefficient"].append(nx.clustering(G))

        try:
            communities = list(nx.algorithms.community.label_propagation_communities(G))
            mod_val = nx.algorithms.community.modularity(G, communities)
        except Exception:
            mod_val = None
        metrics["modularity"].append(mod_val)

        metrics["transitivity"].append(nx.transitivity(G))
        metrics["density"].append(nx.density(G))
        metrics["efficiency"].append(nx.global_efficiency(G))

    return metrics


def get_subj_group(subj_id):
    # 1 for SN, 0 for WSN
    return 1 if subj_id.startswith("SN_") else 0


def rfe_feature_selection(X_train, y_train, n_features, rf_params):
    # Recursive feature elimination with a random forest
    rf = RandomForestClassifier(random_state=random_state, **rf_params)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=10)
    rfe.fit(X_train, y_train)
    return X_train[:, rfe.support_], rfe


def build_design_matrix(metrics_by_subj, subject_subset):
    # One row per epoch. Node level metrics are averaged over the channel
    # groups. SN = 1 and WSN = 0.
    feature_names = []
    for b in bands:
        for gm in global_metrics:
            feature_names.append(f"{b}_{gm}")
        for lm in local_metrics:
            for reg in channel_groups:
                feature_names.append(f"{b}_{lm}_{reg}")

    X_rows = []
    y_labels = []
    subj_id_list = []

    for subj_id, band_dict in metrics_by_subj.items():
        if subj_id not in subject_subset:
            continue

        label = 1 if subj_id.startswith("SN_") else 0

        # Smallest epoch count across the bands
        epoch_counts = []
        for b in bands:
            metric_dict = band_dict[b]
            for m in global_metrics + local_metrics:
                epoch_counts.append(len(metric_dict[m]))
        n_epochs = min(epoch_counts)

        for ep_i in range(n_epochs):
            row_feats = []

            for b in bands:
                metric_dict = band_dict[b]

                for gm in global_metrics:
                    # modularity can be None
                    tmp = metric_dict[gm][ep_i]
                    val = tmp if isinstance(tmp, (int, float)) and np.isfinite(tmp) else np.nan
                    row_feats.append(val)

                for lm in local_metrics:
                    epoch_dict = metric_dict[lm][ep_i]
                    if not isinstance(epoch_dict, dict):
                        row_feats.extend([np.nan] * len(channel_groups))
                    else:
                        for reg_chs in channel_groups.values():
                            region_vals = []
                            for ch_name in reg_chs:
                                ch_idx = channel_labels.index(ch_name)
                                node_val = epoch_dict.get(ch_idx, np.nan)
                                if isinstance(node_val, (int, float)) and np.isfinite(node_val):
                                    region_vals.append(node_val)
                            row_feats.append(np.mean(region_vals) if region_vals else np.nan)

            X_rows.append(row_feats)
            y_labels.append(label)
            subj_id_list.append(subj_id)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_labels, dtype=int)
    subj_ids = np.array(subj_id_list, dtype=object)
    return X, y, subj_ids, feature_names


if __name__ == "__main__":

    print("Loading coherence data")
    rest_data_by_subj = load_and_group_by_subject(
        rest_paths, SN_subject_ids, WSN_subject_ids
    )

    all_subjects = sorted(rest_data_by_subj.keys())
    print(f"{len(all_subjects)} subjects\n")

    # Leave one participant out
    epochwise_results = []
    n_subjects = len(all_subjects)

    for fold_idx in range(n_subjects):
        test_subj = all_subjects[fold_idx]
        train_subjs = [s for s in all_subjects if s != test_subj]

        print(f"fold {fold_idx+1} of {n_subjects}, test subject {test_subj}")

        thresholds = compute_otsu_thresholds_for_training(rest_data_by_subj, train_subjs)
        binarized = binarize_for_subjects(rest_data_by_subj, all_subjects, thresholds)

        metrics_by_subj = {}
        for sid in all_subjects:
            metrics_by_subj[sid] = {}
            for band_name, arr in binarized[sid].items():
                graphs = construct_graphs_for_one_subject(arr)
                metrics_by_subj[sid][band_name] = calculate_graph_metrics_for_one_subject(graphs)

        X_train, y_train, subj_train, feat_names = build_design_matrix(metrics_by_subj, train_subjs)
        X_test,  y_test,  subj_test,  _ = build_design_matrix(metrics_by_subj, [test_subj])

        imputer = SimpleImputer(strategy="mean")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp  = imputer.transform(X_test)

        gkf = GroupKFold(n_splits=5)
        rsearch = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_distributions=rf_param_dist,
            n_iter=10,
            scoring="accuracy",
            cv=gkf,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        rsearch.fit(X_train_imp, y_train, groups=subj_train)
        best_params_rf = rsearch.best_params_

        tuned_rf = RandomForestClassifier(random_state=random_state, **best_params_rf)

        X_train_rfe, rfe_obj = rfe_feature_selection(
            X_train_imp, y_train, n_features_to_select, best_params_rf
        )
        X_test_rfe = X_test_imp[:, rfe_obj.support_]

        # Weight epochs by the inverse of the subject epoch count
        unique_subjs_train, counts = np.unique(subj_train, return_counts=True)
        subj2count = dict(zip(unique_subjs_train, counts))
        train_weights = np.array([1.0 / subj2count[sid] for sid in subj_train])

        models_for_fold = {
            "RandomForest": tuned_rf,
            "SVM (RBF)":    other_models["SVM (RBF)"],
            "GradientBoost": other_models["GradientBoost"],
            "LDA":           other_models["LDA"],
            "NaiveBayes":    other_models["NaiveBayes"]
        }

        test_subj_group = get_subj_group(test_subj)

        for m_name, clf in models_for_fold.items():
            if "sample_weight" in inspect.signature(clf.fit).parameters:
                clf.fit(X_train_rfe, y_train, sample_weight=train_weights)
            else:
                clf.fit(X_train_rfe, y_train)

            y_pred = clf.predict(X_test_rfe)

            # Probability of SN, which is label 1
            sn_idx = np.where(clf.classes_ == 1)[0][0]
            proba_sn = clf.predict_proba(X_test_rfe)[:, sn_idx]

            for true_lbl, pred_lbl, p_sn in zip(y_test, y_pred, proba_sn):
                epochwise_results.append({
                    "subject":   test_subj,
                    "model":     m_name,
                    "y_true":    true_lbl,       # 1 = SN, 0 = WSN
                    "y_pred":    pred_lbl,
                    "proba_SN":  float(p_sn),
                    "group":     test_subj_group
                })

    # One row per subject per model
    df_epoch = pd.DataFrame(epochwise_results)

    all_rows = []

    for (m_name, s_id), sub_df in df_epoch.groupby(["model", "subject"]):
        cm = confusion_matrix(sub_df["y_true"], sub_df["y_pred"], labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        acc = (tp + tn) / (tp + tn + fp + fn)

        all_rows.append({
            "subject":          s_id,
            "model":            m_name,
            "group":            sub_df["group"].iloc[0],
            "p_hat_SN":         np.nanmean(sub_df["proba_SN"].values),
            "sensitivity":      sensitivity,
            "specificity":      specificity,
            "accuracy":         acc
        })

    df_lopo_rest = pd.DataFrame(all_rows)

    print("\nPer subject metrics, mean and SD")
    summary_per_model = {}
    for m_name in df_lopo_rest["model"].unique():
        mask = (df_lopo_rest["model"] == m_name)
        summary_per_model[m_name] = {
            "mean_accuracy":    df_lopo_rest.loc[mask, "accuracy"].mean(),
            "std_accuracy":     df_lopo_rest.loc[mask, "accuracy"].std(ddof=1),
            "mean_sensitivity": df_lopo_rest.loc[mask, "sensitivity"].mean(),
            "std_sensitivity":  df_lopo_rest.loc[mask, "sensitivity"].std(ddof=1),
            "mean_specificity": df_lopo_rest.loc[mask, "specificity"].mean(),
            "std_specificity":  df_lopo_rest.loc[mask, "specificity"].std(ddof=1)
        }
        s = summary_per_model[m_name]

        print(f"\n{m_name}")
        print(f"  accuracy     {s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f}")
        print(f"  sensitivity  {s['mean_sensitivity']:.3f} ± {s['std_sensitivity']:.3f}")
        print(f"  specificity  {s['mean_specificity']:.3f} ± {s['std_specificity']:.3f}")


    def auc_with_ci(y_true, scores, rng, n_boot=2000, alpha=0.05):
        # AUC plus a bootstrap confidence interval
        auc_full = roc_auc_score(y_true, scores)
        n = len(y_true)
        auc_boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            auc_boot.append(roc_auc_score(y_true[idx], scores[idx]))
        auc_boot = np.array(auc_boot)
        return auc_full, np.quantile(auc_boot, alpha / 2), np.quantile(auc_boot, 1 - alpha / 2)


    def ppv_npv(y_true, scores):
        # SN is the positive class, threshold at 0.5
        y_pred = (scores >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        return ppv, npv


    rng = np.random.default_rng(random_state)

    # Subject level, saved but not printed
    clinical_rows = []

    for m_name, sub_df in df_lopo_rest.groupby("model"):
        y_true_sn = (sub_df["group"].values == 1).astype(int)
        scores_sn = sub_df["p_hat_SN"].values
        n_subj = len(sub_df)

        auc_full, auc_low, auc_high = auc_with_ci(y_true_sn, scores_sn, rng)
        ppv_sn, npv_sn = ppv_npv(y_true_sn, scores_sn)
        summ = summary_per_model[m_name]

        clinical_rows.append({
            "model": m_name,
            "n_subjects": n_subj,
            "prevalence_SN": y_true_sn.mean(),
            "PPV_SN_0.50": ppv_sn,
            "NPV_SN_0.50": npv_sn,
            "AUC_SN": auc_full,
            "AUC_SN_CI_low": auc_low,
            "AUC_SN_CI_high": auc_high,
            **summ
        })

    df_clinical = pd.DataFrame(clinical_rows)

    # Epoch level, these are the numbers in the paper
    print("\nEpoch level, SN as the positive class")
    clinical_rows_epoch = []

    for m_name, sub_df in df_epoch.groupby("model"):
        y_true_sn_ep = (sub_df["group"].values == 1).astype(int)
        scores_sn_ep = sub_df["proba_SN"].values
        n_ep = len(sub_df)

        auc_ep, auc_low_ep, auc_high_ep = auc_with_ci(y_true_sn_ep, scores_sn_ep, rng)
        ppv_ep, npv_ep = ppv_npv(y_true_sn_ep, scores_sn_ep)

        print(f"\n{m_name}, {n_ep} epochs")
        print(f"  SN prevalence  {y_true_sn_ep.mean():.3f}")
        print(f"  PPV            {ppv_ep:.3f}")
        print(f"  NPV            {npv_ep:.3f}")
        print(f"  AUC            {auc_ep:.3f}  95% CI {auc_low_ep:.3f} to {auc_high_ep:.3f}")

        clinical_rows_epoch.append({
            "model": m_name,
            "n_epochs": n_ep,
            "prevalence_SN_epoch": y_true_sn_ep.mean(),
            "PPV_SN_0.50_epoch": ppv_ep,
            "NPV_SN_0.50_epoch": npv_ep,
            "AUC_SN_epoch": auc_ep,
            "AUC_SN_epoch_CI_low": auc_low_ep,
            "AUC_SN_epoch_CI_high": auc_high_ep
        })

    df_clinical_epoch = pd.DataFrame(clinical_rows_epoch)

    # Shuffle the subject labels and recompute the AUC to get a null
    print("\nPermutation test")
    N_PERM = 1000
    rng_perm = np.random.default_rng(random_state + 123)

    perm_results = {}
    for target in ["GradientBoost"]:
        df_m = df_lopo_rest[df_lopo_rest["model"] == target]
        y_true_sn = (df_m["group"].values == 1).astype(int)
        scores_sn = df_m["p_hat_SN"].values

        observed_auc = roc_auc_score(y_true_sn, scores_sn)
        null_aucs = np.array([
            roc_auc_score(rng_perm.permutation(y_true_sn), scores_sn)
            for _ in range(N_PERM)
        ])

        p_perm = (np.sum(null_aucs >= observed_auc) + 1) / (N_PERM + 1)

        perm_results[target] = {
            "observed_auc": float(observed_auc),
            "null_aucs": null_aucs,
            "p_perm": float(p_perm),
            "n_valid_permutations": N_PERM
        }

        print(f"\n{target}, {N_PERM} permutations")
        print(f"  null AUC      {null_aucs.mean():.3f} ± {null_aucs.std(ddof=1):.3f}")
        print(f"  p             {p_perm:.3f}")

    # Graph metrics with the thresholds pooled over all subjects
    print("\nComputing metrics on all subjects for the figures")
    thresholds_all = compute_otsu_thresholds_for_training(rest_data_by_subj, all_subjects)
    binarized_all = binarize_for_subjects(rest_data_by_subj, all_subjects, thresholds_all)

    metrics_all = {}
    for sid in all_subjects:
        metrics_all[sid] = {}
        for band_name, arr in binarized_all[sid].items():
            graphs = construct_graphs_for_one_subject(arr)
            metrics_all[sid][band_name] = calculate_graph_metrics_for_one_subject(graphs)

    metrics_path = os.path.join(SAVE_PATH, "rest_all_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics_all, f)
    print(f"saved {metrics_path}")

    lopo_path = os.path.join(SAVE_PATH, "lopo_results_rest_nested.pkl")
    with open(lopo_path, "wb") as f:
        pickle.dump(
            {
                "df_epoch": df_epoch,
                "df_lopo_rest": df_lopo_rest,
                "df_clinical": df_clinical,
                "df_clinical_epoch": df_clinical_epoch,
                "perm_results": perm_results,
            },
            f,
        )
    print(f"\nsaved {lopo_path}")
