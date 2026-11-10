# Preprocessing notes

The scripts here import the CSV, filter, correct the trigger timing, and
cut epochs around each target. The cleaning was done by hand in EEGLAB,
then a second script splits the trials into ipsi and contra.

MATLAB 2022a, EEGLAB 2023.1.

## Files

- `Preprocess_task_EEG.m` - reads the CSV, filters, builds events, cuts epochs
- `tskpreprocessing.m` - run this. Edit the six variables at the top first.
- `Save_Preprocessed_task_EEG.m` - splits ipsi and contra into separate files
- `chanlocs.mat` - channel locations for the 16-channel montage

Channel order:

    Fp1 Fp2 F3 F4 Fz FC1 FC2 Cz P1 P2 C1 C2 CP3 CP4 O1 O2

Four csv files need to be in the data folder:

    data_received.csv      EEG in columns 1-16, trigger in column 17
    PC_Elapsed.csv         time the trigger was sent, ms
    HoloLens_Elapsed.csv   time the target appeared, ms
    Targets_Order.csv      target ID per trial

The trigger column holds the trial index 1-216 at each target onset. The
script does not use it because it does not account for the Bluetooth delay,
the corrected timing is built from PC_Elapsed and HoloLens_Elapsed instead.
The raw onsets are still useful as a check, the corrected samples should
land within a few tens of milliseconds of them.

## How to run one recording

1. Edit the six variables at the top of `tskpreprocessing.m`. `out_ipsi`
   and `out_contra` only change when you switch between SN and WSN.
2. Run the first two cells. This writes `<set_name>_Epoched.set` into
   the data folder.
3. Run `eeglab` and load that .set file.
4. Do the manual steps below.
5. Make a `preprocessed/new` subfolder inside the data folder and save there
   as `preprocessed_<set_name>.set`. The name and location matter -
   `Save_Preprocessed_task_EEG.m` looks for exactly that. Note this is
   `preprocessed/new`, not `preprocessed` like the resting pipeline.
6. Run the last cell. This splits the file into `_ipsi` and `_contra`
   versions and saves them into `out_ipsi` and `out_contra`, which are the
   folders the connectivity code reads from.

## What the script does before EEGLAB

Bandpass 1-40 Hz with `pop_eegfiltnew`, on the continuous data before
epoching.

Trigger correction. The HoloLens sends target onset over Bluetooth, so the
trigger recorded on the PC lags the actual target appearance. The correction
uses a fixed 75 ms two-way delay, halved, plus the difference between the
HoloLens and PC elapsed times.

Trial labels. Ipsilesional means the target appeared on the same side as the
lesion, contralesional means the opposite side, which is the neglected side.
For a right hemisphere lesion the contralesional field is the left one, so
the left half of the 12-column grid gets `contra` and the right half gets
`ipsi`. This assumes a right hemisphere lesion. Lesion side is handled in the
connectivity code, so at this stage the labels are really just left and
right.

Epochs are -200 to +800 ms around target onset, baseline corrected on the
pre-onset interval.

## Manual steps in EEGLAB

In this order:

1. Scrolled through epochs with `pop_eegplot` and rejected any with obvious
   large-amplitude artifact.
2. Marked bad channels and interpolated: `pop_interp(EEG, [...], 'spherical')`.
3. Re-referenced to average: `pop_reref(EEG, [])`.
4. ICA: `pop_runica(EEG, 'icatype','runica', 'extended',1, 'pca',N)`.
   N = 16 - (number interpolated) - 1, which is the rank of the data after
   interpolation and average referencing. For a session with 4 interpolated
   channels, N = 11.
5. `pop_iclabel(EEG, 'default')`, then removed flagged components by hand.
   Not everything ICLabel flagged - it was a judgment call per session. In
   some sessions ICLabel found nothing and no components were removed.

Steps 1 and 5 were subjective, which is why this stage can't be scripted
exactly. Steps 2, 3, 4 are fully specified above.

No filtering step here, the script already did it. Otherwise the order is the
same as resting state.

## Filtering applied before these scripts

The 2-62 Hz Butterworth bandpass and 58-62 Hz notch reported in the paper
were applied during recording, not here. The CSV is already filtered. The
1-40 Hz bandpass is in `Preprocess_task_EEG.m`.

## Sessions

Up to two sessions per participant, one task recording each. No WH/WOH split
because the headset is on for the whole task recording.

Each recording is split into ipsi and contra files at the end, so one session
produces two .mat files. The saved variables are named `EEG_ipsi` and
`EEG_contra`, not `EEG` like the resting files.

The two conditions stay separate through the connectivity stage because they
were compared against each other first. Wilcoxon signed-rank tests found no
significant differences between them and classification performance was
similar, so they were merged for the classification analyses in the paper.
The split files are still needed to reproduce that comparison.
