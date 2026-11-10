function Preprocess_EEG(data_path, subject_ID, Session, set_name)
    % Read csv file
    data = csvread([data_path,'/data_received.csv'], 1, 0);
    % Data organization
    data = data(:,1:16)';
    fs = 256; % Sampling rate
    % Creating EEG dataset
    EEG.data = data;
    EEG.srate = fs;
    EEG.setname = set_name; 
    EEG.filename = set_name;
    EEG.filepath = data_path;
    EEG.subject = subject_ID;  
    EEG.session = Session;     
    EEG.comments = 'Resting state EEG';  
    EEG.nbchan = size(EEG.data, 1);
    EEG.pnts = size(EEG.data, 2);
    EEG.trials = 1;  
    EEG.xmin = 0;    
    EEG.xmax = (EEG.pnts-1)/EEG.srate; 
    EEG.icasphere = [];
    EEG.icaweights = [];
    EEG.icawinv = [];
    EEG.icaact = [];
    EEG.icachansind = [];
    
    % Load channel locations 
    load(fullfile(fileparts(mfilename('fullpath')), 'chanlocs.mat'));
    EEG.chanlocs = chanlocs;
    EEG.chaninfo = chaninfo;

    % Create events every 2 seconds
    EEG.event = [];
    for i = 1:(2 * EEG.srate):size(EEG.data,2)  
        EEG.event(end+1).type = '2s';    
        EEG.event(end).latency = i;     
    end
    % Check, segment and save the dataset
    EEG = eeg_checkset(EEG, 'eventconsistency'); 
    EEG = pop_epoch(EEG, {'2s'}, [0 2], 'newname', [set_name, '_2secEpochs'], 'epochinfo', 'yes');
    EEG = pop_saveset(EEG, 'filename', [set_name, '_2secEpochs'], 'filepath', data_path);
end
