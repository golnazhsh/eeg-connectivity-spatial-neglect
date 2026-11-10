clear
clc
%% load data info
% Folders holding the split ipsi and contra mat files
path_SN_ipsi    = '<folder with the SN ipsi mat files>';
path_SN_contra  = '<folder with the SN contra mat files>';
path_WSN_ipsi   = '<folder with the WSN ipsi mat files>';
path_WSN_contra = '<folder with the WSN contra mat files>';

% Existing files are named newpreprocessed_, the current preprocessing
% code writes preprocessed_
SN_prefix  = 'newpreprocessed_EEG_SN';
WSN_prefix = 'newpreprocessed_EEG_WSN';

% Sessions that went into the paper, as <subject>_<session>, in the order
% the Python code expects
SN_sessions = {'101_1', '101_2', '102_1', '102_2', '103_1', '103_2', ...
    '104_1', '104_2', '105_2', '106_1', '106_2', '112_2', '113_1', ...
    '114_1', '114_2', '115_2', '116_1', '116_2', '117_2'};

WSN_sessions = {'101_1', '101_2', '102_1', '103_2', '104_1', '104_2', ...
    '108_1', '108_2', '110_1', '110_2', '115_1', '115_2', '116_1', ...
    '117_1', '118_1', '119_1'};

% Left hemisphere lesions, electrodes get mirrored so the lesion side is
% on the right for everyone
SN_mirror  = {'104_1', '104_2', '117_2'};
WSN_mirror = {'101_1', '101_2', '102_1', '104_1', '104_2', '116_1'};
%% Coherence
Coherence_task_EEG(path_SN_contra,  SN_prefix,  SN_sessions,  SN_mirror,  'contra', 'SN_contra');
Coherence_task_EEG(path_SN_ipsi,    SN_prefix,  SN_sessions,  SN_mirror,  'ipsi',   'SN_ipsi');
Coherence_task_EEG(path_WSN_contra, WSN_prefix, WSN_sessions, WSN_mirror, 'contra', 'WSN_contra');
Coherence_task_EEG(path_WSN_ipsi,   WSN_prefix, WSN_sessions, WSN_mirror, 'ipsi',   'WSN_ipsi');
