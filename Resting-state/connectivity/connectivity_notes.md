# Connectivity notes

Coherence between every channel pair, per epoch, in four frequency bands.
Takes the preprocessed mat files and writes one file per group per band.

MATLAB 2022a. The mat files were computed on 2022a. Running the same
code on R2025a gives differences around 5e-7, which is single precision
rounding, not a change in the analysis.

## Files

- `Coherence_EEG.m` - loads the sessions, mirrors, computes coherence, saves
- `rsConnectivity.m` - run this. Edit the two paths at the top first.
- `band_specific_coherence.m` - coherence for one channel pair

## How to run

1. Edit the two paths at the top of `rsConnectivity.m`. These are the
   `out_path` folders the preprocessing scripts wrote the mat files into.
2. Run it. It prints a line per session, 36 in total, and is slow.
3. The eight mat files are written to the current folder, which is where
   the Python code reads them from.

## Sessions

23 SN sessions from 10 participants and 13 WSN sessions from 6, listed in
`rsConnectivity.m`. Sessions left out were the lower quality ones, judged on
ICA artifact proportion, interpolated channel count, usable epochs, and
residual noise.

The order of the two lists matters. The Python code has its own subject list
that it matches to these by position, so reordering them would silently give
every session the wrong participant label.

## Mirroring

SN117 and WSN116 have left hemisphere lesions. Their electrodes are swapped
left to right so the lesioned hemisphere is on the right for everyone:

    swap_idx = [2 1 4 3 5 7 6 8 10 9 12 11 14 13 16 15]

Fp1 and Fp2 swap, F3 and F4, FC1 and FC2, C1 and C2, P1 and P2, CP3 and CP4,
O1 and O2. Fz and Cz stay where they are.

## Coherence settings

`mscohere` with a 64 sample Hamming window, 50 percent overlap, nfft 128.
At 256 Hz that puts a frequency bin every 2 Hz.

    delta  1-4 Hz    averages 1 bin
    theta  4-8 Hz    averages 2 bins
    alpha  8-13 Hz   averages 3 bins
    beta   13-30 Hz  averages 8 bins

Delta comes down to a single bin at 2 Hz, and the 64 sample window is 0.25 s,
so the real frequency resolution is 4 Hz, wider than the delta band itself.
Delta is the least reliable of the four. The paper's main findings are in
beta and theta.

The paper lists theta as 4-7 and alpha as 8-12. With 2 Hz bins theta picks
the same two bins either way, alpha may differ by the 12 Hz bin.

## Output

Each file holds one cell array, `group_<band>Coh`, with one cell per session
and each cell 16 x 16 x number of epochs. The diagonal is zero, the matrix
is symmetric.

## Data

8 mat files, 4 SN and 4 WSN, 1 per frequency band. These are the coherence
matrices, not the EEG itself. Use them as the baseline to compare new
results against.
