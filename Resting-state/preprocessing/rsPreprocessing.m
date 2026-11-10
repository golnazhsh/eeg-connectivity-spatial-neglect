clear
clc
%% load data info
% Changes per recording. out_path only changes between SN and WSN.
% set_name: rsEEG_<subject><session>_<WH|WOH>
%   WH  = headset on, recorded first, folder RestEEG_<timestamp>_w_Hololens
%   WOH = headset off, recorded last, folder RestEEG_<timestamp>_wo_Hololens
data_path  = '<session folder>/RestEEG_<timestamp>_w_Hololens';
subject_ID = 'SN_000';
Session    = 1;
set_name   = 'rsEEG_SN000_1_WH';
out_path   = '<folder collecting all the SN mat files>';
%% Preprocessing
Preprocess_EEG(data_path, subject_ID, Session, set_name);
%% EEGLab
eeglab
%% Save the Preprocessed EEG
Save_Preprocessed_EEG(data_path, set_name, out_path)
