function Coherence_EEG(data_path, file_list, mirror_list, group_name)
    % Left and right electrode swap, for left hemisphere lesions
    swap_idx = [2, 1, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15];

    fs = 256; % Sampling rate
    n_sessions = length(file_list);

    group_deltaCoh = cell(1, n_sessions);
    group_thetaCoh = cell(1, n_sessions);
    group_alphaCoh = cell(1, n_sessions);
    group_betaCoh  = cell(1, n_sessions);

    for s = 1:n_sessions
        set_name = file_list{s};
        loaded = load(fullfile(data_path, set_name));
        data = loaded.EEG.data;

        if any(strcmp(set_name, mirror_list))
            data = data(swap_idx, :, :);
        end

        nChannels = size(data, 1);
        nEpochs = size(data, 3);

        deltaCoh = zeros(nChannels, nChannels, nEpochs);
        thetaCoh = zeros(nChannels, nChannels, nEpochs);
        alphaCoh = zeros(nChannels, nChannels, nEpochs);
        betaCoh  = zeros(nChannels, nChannels, nEpochs);

        % Coherence for every channel pair, diagonal stays zero
        for epoch = 1:nEpochs
            for j = 1:nChannels
                for k = j+1:nChannels
                    [d, t, a, b] = band_specific_coherence(data(j, :, epoch), data(k, :, epoch), fs);
                    deltaCoh(j, k, epoch) = d;
                    thetaCoh(j, k, epoch) = t;
                    alphaCoh(j, k, epoch) = a;
                    betaCoh(j, k, epoch)  = b;
                    deltaCoh(k, j, epoch) = d;
                    thetaCoh(k, j, epoch) = t;
                    alphaCoh(k, j, epoch) = a;
                    betaCoh(k, j, epoch)  = b;
                end
            end
        end

        group_deltaCoh{s} = deltaCoh;
        group_thetaCoh{s} = thetaCoh;
        group_alphaCoh{s} = alphaCoh;
        group_betaCoh{s}  = betaCoh;

        fprintf('%d of %d  %s  %d epochs\n', s, n_sessions, set_name, nEpochs);
    end

    % Saves to the current folder
    save([group_name, '_deltaCoh.mat'], 'group_deltaCoh');
    save([group_name, '_thetaCoh.mat'], 'group_thetaCoh');
    save([group_name, '_alphaCoh.mat'], 'group_alphaCoh');
    save([group_name, '_betaCoh.mat'], 'group_betaCoh');
end