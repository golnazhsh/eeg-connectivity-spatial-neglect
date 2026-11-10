# Preprocessing notes

The scripts here import the CSV and cut 2 s epochs. Everything after that
was done by hand in the EEGLAB.

MATLAB 2022a, EEGLAB 2023.1.

## Files

- `Preprocess_EEG.m` - reads the CSV, builds the EEG struct, cuts epochs
- `rsPreprocessing.m` - run this. Edit the five variables at the top first.
- `Save_Preprocessed_EEG.m` - converts the finished .set to .mat
- `chanlocs.mat` - channel locations for the 16-channel montage

Channel order:

    Fp1 Fp2 F3 F4 Fz FC1 FC2 Cz P1 P2 C1 C2 CP3 CP4 O1 O2

`data_received.csv` has 17 columns. Columns 1-16 are the EEG channels in
the order above, column 17 is the trigger, all zeros here since there are
no events during resting state.

## How to run one recording

1. Edit the five variables at the top of `rsPreprocessing.m`. `out_path`
   only changes when you switch between SN and WSN.
2. Run the first two cells. This writes `<set_name>_2secEpochs.set` into
   the data folder.
3. Run `eeglab` and load that .set file.
4. Do the manual steps below.
5. Make a `preprocessed` subfolder inside the data folder and save there as
   `preprocessed_<set_name>.set`. The name and location matter -
   `Save_Preprocessed_EEG.m` looks for exactly that.
6. Run the last cell. This writes the .mat into `out_path`, which is the
   folder the connectivity code reads from.

## Manual steps in EEGLAB

In this order:

1. Removed the first epoch (onset artifact).
2. Bandpass filter: `pop_eegfiltnew(EEG, 'locutoff',1, 'hicutoff',30)`.
3. Scrolled through epochs with `pop_eegplot` and rejected any with obvious
   large-amplitude artifact.
4. Marked bad channels and interpolated: `pop_interp(EEG, [...], 'spherical')`.
5. Re-referenced to average: `pop_reref(EEG, [])`.
6. ICA: `pop_runica(EEG, 'icatype','runica', 'extended',1, 'pca',N)`.
   N = 16 - (number interpolated) - 1, which is the rank of the data after
   interpolation and average referencing. For a session with 4 interpolated
   channels, N = 11.
7. `pop_iclabel(EEG, 'default')`, then removed flagged components by hand.
   Not everything ICLabel flagged - it was a judgment call per session. In
   some sessions ICLabel found nothing and no components were removed.

Steps 3 and 7 were subjective, which is why this stage can't be scripted
exactly. Steps 2, 4, 5, 6 are fully specified above.

Note the order: interpolation came after the epoch rejection pass, not
before.

## Filtering applied before these scripts

The 2-62 Hz Butterworth bandpass and 58-62 Hz notch reported in the paper
were applied during recording, not here. The CSV is already filtered.

## Sessions

Two sessions per participant, two resting recordings each:

- WH = headset on, recorded first (before clicker calibration)
- WOH = headset off, recorded last (after the task)

These are the pre- and post-task recordings in the paper. They are not the
same condition - WOH has no AR fixation cross because the headset is off.
Both are pooled in the analysis.
