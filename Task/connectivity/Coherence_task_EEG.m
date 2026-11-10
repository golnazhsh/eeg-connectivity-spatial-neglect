function Coherence_task_EEG(data_path, file_prefix, sessions, mirror_sessions, condition, group_name)
    % Left and right electrode swap, for left hemisphere lesions
    swap_idx = [2, 1, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15];

    fs = 256; % Sampling rate
    n_sessions = length(sessions);
    var_name = ['EEG_', condition];

    group_deltaCoh = cell(1, n_sessions);
    group_thetaCoh = cell(1, n_sessions);
    group_alphaCoh = cell(1, n_sessions);
    group_betaCoh  = cell(1, n_sessions);
    group_gammaCoh = cell(1, n_sessions);

    for s = 1:n_sessions
        set_name = sprintf('%s%s_%s.mat', file_prefix, sessions{s}, condition);
        loaded = load(fullfile(data_path, set_name));
        data = loaded.(var_name).data;

        if ismember(sessions{s}, mirror_sessions)
            data = data(swap_idx, :, :);
        end

        nChannels = size(data, 1);
        nEpochs = size(data, 3);

        deltaCoh = zeros(nChannels, nChannels, nEpochs);
        thetaCoh = zeros(nChannels, nChannels, nEpochs);
        alphaCoh = zeros(nChannels, nChannels, nEpochs);
        betaCoh  = zeros(nChannels, nChannels, nEpochs);
        gammaCoh = zeros(nChannels, nChannels, nEpochs);

        % Coherence for every channel pair, diagonal stays zero
        for epoch = 1:nEpochs
            for j = 1:nChannels
                for k = j+1:nChannels
                    [d, t, a, b, g] = band_specific_coherence(data(j, :, epoch), data(k, :, epoch), fs);
                    deltaCoh(j, k, epoch) = d;
                    thetaCoh(j, k, epoch) = t;
                    alphaCoh(j, k, epoch) = a;
                    betaCoh(j, k, epoch)  = b;
                    gammaCoh(j, k, epoch) = g;
                    deltaCoh(k, j, epoch) = d;
                    thetaCoh(k, j, epoch) = t;
                    alphaCoh(k, j, epoch) = a;
                    betaCoh(k, j, epoch)  = b;
                    gammaCoh(k, j, epoch) = g;
                end
            end
        end

        group_deltaCoh{s} = deltaCoh;
        group_thetaCoh{s} = thetaCoh;
        group_alphaCoh{s} = alphaCoh;
        group_betaCoh{s}  = betaCoh;
        group_gammaCoh{s} = gammaCoh;

        fprintf('%s  %d of %d  %s  %d epochs\n', group_name, s, n_sessions, set_name, nEpochs);
    end

    % Saves to the current folder
    save([group_name, '_deltaCoh_new.mat'], 'group_deltaCoh');
    save([group_name, '_thetaCoh_new.mat'], 'group_thetaCoh');
    save([group_name, '_alphaCoh_new.mat'], 'group_alphaCoh');
    save([group_name, '_betaCoh_new.mat'],  'group_betaCoh');
    save([group_name, '_gammaCoh_new.mat'], 'group_gammaCoh');
end
