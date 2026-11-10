# Main analysis notes

Needs numpy, scipy, pandas, networkx, scikit-image, scikit-learn, mne,
statsmodels.

## Files

- `final_rest.py` - leave one participant out classification
- `topomaps_rest.py` - topographic maps of Cohen's d between the groups

## How to run

Set `data_path` at the top of `final_rest.py` to the folder with the
coherence mat files. Run `final_rest.py`, then `topomaps_rest.py`.

`final_rest.py` writes `lopo_results_rest_nested.pkl` and
`rest_all_metrics.pkl`. The topomap script reads the second one.

## final_rest.py

IQR epoch outlier removal, factor 0.75, per subject and band. Then for each
held out participant:

- Otsu threshold per band from the training subjects
- binarize, build graphs, compute the 8 metrics
- node level metrics averaged over the channel groups
- tune the random forest, RandomizedSearchCV with GroupKFold
- RFE down to 30 features
- fit random forest, SVM, gradient boosting, LDA, naive Bayes
- epochs weighted by the inverse of the subject epoch count

Then a permutation test on the subject level AUC, 1000 shuffles.

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

    2491 epochs

    GradientBoost   accuracy 0.868   sensitivity 0.950   specificity 0.732
    RandomForest    accuracy 0.868   sensitivity 0.964   specificity 0.710
    LDA             accuracy 0.804   sensitivity 0.917   specificity 0.617
    NaiveBayes      accuracy 0.708   sensitivity 0.842   specificity 0.485
    SVM (RBF)       accuracy 0.768   sensitivity 0.877   specificity 0.586

    GradientBoost epoch level AUC 0.902, RandomForest 0.926
    permutation null AUC 0.503 ± 0.158, p 0.001

## topomaps_rest.py

Mann Whitney U per channel, FDR corrected across the 16 channels within each
metric and band. Global metrics get one FDR over all metric and band pairs.

Cohen's d is WSN minus SN, so negative means higher in SN.

Eigenvector centrality at C1 in beta, d -1.971. Betweenness centrality
significant at frontal and central channels in delta, theta and beta, d
from -1.030 to -2.584.
