# Main analysis notes

Needs numpy, scipy, pandas, networkx, scikit-image, scikit-learn, mne,
statsmodels.

## Files

- `final_task.py` - leave one participant out classification
- `topomaps_task.py` - topographic maps of Cohen's d between the groups

## How to run

Set `data_path` at the top of `final_task.py` to the folder with the
coherence mat files. Run `final_task.py`, then `topomaps_task.py`.

`final_task.py` writes `lopo_results_task_merged_nested_with_perm.pkl` and
`task_all_metrics.pkl`. The topomap script reads the second one.

## final_task.py

Ipsi and contra are loaded separately and the design matrices are stacked
before fitting.

IQR epoch outlier removal, factor 0.75, per subject, band and condition.
Then for each held out participant, and for ipsi and contra separately:

- Otsu threshold per band from the training subjects
- binarize, build graphs, compute the 8 metrics
- node level metrics averaged over the channel groups

The two design matrices are stacked, then:

- tune the random forest, RandomizedSearchCV with GroupKFold
- RFE down to 30 features
- fit random forest, SVM, gradient boosting, LDA, naive Bayes
- epochs weighted by the inverse of the subject epoch count

Then a permutation test on the random forest subject level AUC, 1000
shuffles.

`random_state = 88`. SN = 1, WSN = 0.

## Channel groups

    L_Frontal   Fp1, F3
    R_Frontal   Fp2, F4
    L_Central   FC1, CP3, C1
    R_Central   FC2, CP4, C2
    L_ParOcc    P1, O1
    R_ParOcc    P2, O2
    Midline     Fz, Cz

## What to expect

    23 subjects, 5977 epochs

    LDA             accuracy 0.809   sensitivity 0.855   specificity 0.758
    RandomForest    accuracy 0.799   sensitivity 0.856   specificity 0.738
    GradientBoost   accuracy 0.797   sensitivity 0.827   specificity 0.764
    NaiveBayes      accuracy 0.768   sensitivity 0.785   specificity 0.750
    SVM (RBF)       accuracy 0.730   sensitivity 0.708   specificity 0.753

    LDA epoch level AUC 0.889, RandomForest 0.880
    permutation null AUC 0.501 ± 0.124, p 0.001

SVM here has `class_weight='balanced'`, which the original script did not.
Its accuracy comes out 0.730 instead of 0.679.

## topomaps_task.py

Ipsi and contra epochs are merged first. Mann Whitney U per channel, FDR
corrected across the 16 channels within each metric and band. Global
metrics get one FDR over all metric and band pairs.

Cohen's d is WSN minus SN, so negative means higher in SN.

Eigenvector centrality at C1: d -2.548 delta, -2.643 theta, -1.956 beta,
-1.845 gamma.

