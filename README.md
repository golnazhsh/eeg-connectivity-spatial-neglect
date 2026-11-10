# AREEN EEG connectivity and neglect classification

EEG preprocessing, coherence connectivity and classification of spatial
neglect from the AREEN system. Resting state and task, analysed separately.

Brain Network Connectivity During Resting-State and a Visuospatial Task as
a Biomarker for Spatial Neglect in Stroke Patients. Neurorehabilitation and
Neural Repair 2026;40(6):482-494. doi:10.1177/15459683261419423

## Layout

Each condition has the same three folders, run in this order.

    Resting-state/           Task/
      preprocessing/           preprocessing/
      connectivity/            connectivity/
      main_analysis/           main_analysis/

Each folder has its own notes file with the settings and what to expect.

### preprocessing

MATLAB, EEGLAB 2023.1. Reads the recording csv, builds an EEGLAB dataset,
epochs it, and saves a mat file per session. Filtering, bad channel
interpolation, re-referencing and ICA are done by hand in the EEGLAB GUI
between the two scripts.

| File | Purpose |
|---|---|
| `rsPreprocessing.m`, `tskpreprocessing.m` | Driver. Edit the paths at the top and run. |
| `Preprocess_EEG.m`, `Preprocess_task_EEG.m` | csv to epoched EEGLAB dataset |
| `Save_Preprocessed_EEG.m`, `Save_Preprocessed_task_EEG.m` | Cleaned dataset to mat |
| `chanlocs.mat` | 16 channel montage |

### connectivity

MATLAB. Magnitude squared coherence per channel pair per epoch, averaged
within frequency bands. Left hemisphere lesion sessions get their
electrodes mirrored.

| File | Purpose |
|---|---|
| `rsConnectivity.m`, `tskConnectivity.m` | Driver, holds the session lists |
| `Coherence_EEG.m`, `Coherence_task_EEG.m` | Coherence per session |
| `band_specific_coherence.m` | Band averaging. Two versions, 4 bands for resting and 5 for task. |

This repository contains code and documentation only.

### main_analysis

Python. Graph metrics from thresholded coherence, then leave one
participant out classification, and a separate group comparison per
electrode.

| File | Purpose |
|---|---|
| `final_rest.py`, `final_task.py` | Classification. Run first. |
| `topomaps_rest.py`, `topomaps_task.py` | Cohen's d between groups per electrode |

## Notes

The session lists in the connectivity drivers are matched by position to
the subject lists in the Python scripts.

The resting and task versions of `band_specific_coherence.m` have the same
name and different outputs. Keep only one on the MATLAB path at a time.
