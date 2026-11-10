# Connectivity notes

Coherence between every channel pair, per epoch, in five frequency bands.
Takes the split ipsi and contra mat files and writes one file per group per
condition per band.

MATLAB 2022a. Running the same code on a newer release gives tiny differences.

## Files

- `Coherence_task_EEG.m` - loads the sessions, mirrors, computes coherence, saves
- `tskConnectivity.m` - run this. Edit the four paths at the top first.
- `band_specific_coherence.m` - coherence for one channel pair

This `band_specific_coherence.m` returns five bands. The resting one returns
four. They have the same name, so whichever is on the MATLAB path wins. Keep
them in their own folders and do not add both.

## How to run

1. Edit the four paths at the top of `tskConnectivity.m`. These are the
   `out_ipsi` and `out_contra` folders the preprocessing scripts wrote the
   mat files into.
2. Check the file prefix. The existing files are named `newpreprocessed_`
   because of an old path bug. Files written by the current version of
   `Save_Preprocessed_task_EEG.m` are named `preprocessed_`.
3. Run it. It prints a line per session, 70 in total, and is slow.
4. The 20 mat files are written to the current folder, which is where the
   Python code reads them from.

## Sessions

19 SN sessions from 12 participants and 16 WSN sessions from 11, listed in
`tskConnectivity.m` as `<subject>_<session>`. Each session contributes both
an ipsi and a contra file.

The order of the two lists matters. The Python code has its own subject list
that it matches to these by position, so reordering them would silently give
every session the wrong participant label.

## Mirroring

SN104, SN117, WSN101, WSN102, WSN104 and WSN116 have left hemisphere
lesions. Their electrodes are swapped left to right so the lesioned
hemisphere is on the right for everyone:

    swap_idx = [2 1 4 3 5 7 6 8 10 9 12 11 14 13 16 15]

Fp1 and Fp2 swap, F3 and F4, FC1 and FC2, C1 and C2, P1 and P2, CP3 and CP4,
O1 and O2. Fz and Cz stay where they are.

This swaps electrodes, not stimulus positions. The ipsi and contra labels
themselves are assigned during preprocessing assuming a right hemisphere
lesion.

## Coherence settings

`mscohere` with a 64 sample Hamming window, 50 percent overlap, nfft 128.
At 256 Hz that puts a frequency bin every 2 Hz.

    delta  1-4 Hz    averages 1 bin
    theta  4-8 Hz    averages 2 bins
    alpha  8-13 Hz   averages 3 bins
    beta   13-30 Hz  averages 8 bins
    gamma  30-40 Hz  averages 6 bins

Delta comes down to a single bin at 2 Hz, and the 64 sample window is 0.25 s,
so the real frequency resolution is 4 Hz, wider than the delta band itself.
Delta is the least reliable of the five.

Gamma uses an inclusive upper bound where the other four are exclusive, so
it picks up the 40 Hz bin. That bin sits at the edge of the 1-40 Hz filter
and is partly attenuated.

Task epochs are 1 s where resting epochs are 2 s. The window is the same
for both, so task coherence comes from about half as many segments, which
biases it upward. Task and resting coherence values are not directly
comparable in magnitude. Group comparisons within a condition are fine.

## Output

Each file holds one cell array, `group_<band>Coh`, with one cell per session
and each cell 16 x 16 x number of epochs. The diagonal is zero, the matrix
is symmetric.

The `_new` on the filenames is what the Python code expects.

## Data

20 mat files, 5 bands for each of SN ipsi, SN contra, WSN ipsi and WSN
contra. These are the coherence matrices, not the EEG itself. Use them as
the baseline to compare new results against.
