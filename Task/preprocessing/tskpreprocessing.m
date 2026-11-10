clear
clc
%% load data info
% Changes per recording. The output folders only change between SN and WSN.
% set_name: EEG_<subject>_<session>
data_path  = '<session folder>/EEGbased_<timestamp>';
subject_ID = 'SN000';
Session    = 1;
set_name   = 'EEG_SN000_1';
out_ipsi   = '<folder collecting all the SN ipsi files>';
out_contra = '<folder collecting all the SN contra files>';
%% Preprocessing
Preprocess_task_EEG(data_path, subject_ID, Session, set_name);
%% EEGLab
eeglab
%% Save the Preprocessed EEG
Save_Preprocessed_task_EEG(data_path, set_name, out_ipsi, out_contra)
