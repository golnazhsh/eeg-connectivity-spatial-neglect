function [deltaCoh, thetaCoh, alphaCoh, betaCoh, gammaCoh] = band_specific_coherence(signal1, signal2, fs, window_length, freq_bands)
if nargin < 4 || isempty(window_length)
    window_length = 64;
end
if nargin < 5 || isempty(freq_bands)
    freq_bands = [1 4; 4 8; 8 13; 13 30; 30 40];
end
window = hamming(window_length);
noverlap = round(0.5 * window_length);
nfft = 128;
[coh, f] = mscohere(signal1, signal2, window, noverlap, nfft, fs);
deltaCoh = mean(coh(f >= freq_bands(1,1) & f <  freq_bands(1,2)));
thetaCoh = mean(coh(f >= freq_bands(2,1) & f <  freq_bands(2,2)));
alphaCoh = mean(coh(f >= freq_bands(3,1) & f <  freq_bands(3,2)));
betaCoh  = mean(coh(f >= freq_bands(4,1) & f <  freq_bands(4,2)));
gammaCoh = mean(coh(f >= freq_bands(5,1) & f <= freq_bands(5,2)));
end