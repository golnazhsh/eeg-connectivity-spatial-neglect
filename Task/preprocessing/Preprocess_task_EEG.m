function Preprocess_task_EEG(data_path, subject_ID, Session, set_name)
    % Read csv file
    data = csvread([data_path, '/data_received.csv'], 1, 0);
    % Data organization
    data = data(:, 1:16)';
    fs = 256; % Sampling rate
    % Creating EEG dataset
    EEG.data = data;
    EEG.srate = fs;
    EEG.setname = set_name;
    EEG.filename = set_name;
    EEG.filepath = data_path;
    EEG.subject = subject_ID;
    EEG.session = Session;
    EEG.nbchan = size(EEG.data, 1);
    EEG.pnts = size(EEG.data, 2);
    EEG.trials = 1;
    EEG.xmin = 0;
    EEG.xmax = (EEG.pnts - 1) / EEG.srate;
    EEG.comments = '';
    EEG.icawinv = [];
    EEG.icasphere = [];
    EEG.icaweights = [];

    % Load channel locations
    load(fullfile(fileparts(mfilename('fullpath')), 'chanlocs.mat'));
    EEG.chanlocs = chanlocs;
    EEG.chaninfo = chaninfo;

    % Needed before filtering
    EEG.event = [];

    EEG = pop_eegfiltnew(EEG, 1, 40);

    % Trigger timing. PC_Elapsed is when the trigger was sent,
    % HoloLens_Elapsed is when the target appeared. Correction from
    % previous codes.
    PC_T = table2array(readtable([data_path, '/PC_Elapsed.csv'], 'ReadVariableNames', false));
    Holo_T = table2array(readtable([data_path, '/HoloLens_Elapsed.csv'], 'ReadVariableNames', true));

    two_way_delay = 75; % in ms
    correction_ms = Holo_T - PC_T + (two_way_delay / 2);
    corrected_trigger_samples = round((PC_T / 1000) * fs + (correction_ms / 1000) * fs);

    disp('Total EEG data points (EEG.pnts):');
    disp(EEG.pnts);

    % Target order, used to label each trial by grid side
    targets_order = table2array(readtable([data_path, '/Targets_Order.csv'], 'ReadVariableNames', true, 'TextType', 'string'));

    % Left half of the 12 column grid is 'contra', right half is 'ipsi'.
    % Assumes a right hemisphere lesion, mirroring is done in the
    % connectivity code.
    for i = 1:length(corrected_trigger_samples)
        target = targets_order(i);
        column = mod(target - 1, 12) + 1; % 1 to 12

        if column >= 1 && column <= 6
            event_type = 'contra';
        elseif column >= 7 && column <= 12
            event_type = 'ipsi';
        else
            event_type = 'undefined_target';
        end

        % Triggers outside the recording
        event_sample = corrected_trigger_samples(i);
        if event_sample > 0 && event_sample <= EEG.pnts
            EEG.event(end+1).type = event_type;
            EEG.event(end).latency = event_sample;
            EEG.event(end).urevent = i;
        else
            warning(['Event ' num2str(i) ' latency ' num2str(event_sample) ' exceeds data range and will be ignored.']);
        end
    end

    % Epoch around target onset
    epoch_window = [-0.2, 0.8]; % in seconds
    event_types = {'contra', 'ipsi'};
    EEG = pop_epoch(EEG, event_types, epoch_window);

    % Baseline from epoch start to target onset
    EEG = pop_rmbase(EEG, [EEG.xmin * 1000, 0]);

    % Check and save the dataset
    EEG = eeg_checkset(EEG, 'eventconsistency');
    pop_saveset(EEG, 'filename', [set_name, '_Epoched.set'], 'filepath', data_path);
end