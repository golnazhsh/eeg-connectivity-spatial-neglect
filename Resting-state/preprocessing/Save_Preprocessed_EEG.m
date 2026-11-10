function Save_Preprocessed_EEG(data_path, set_name, out_path)
    % Load the .set file for preprocessing
    EEG = pop_loadset('filename', ['preprocessed_',set_name, '.set'], 'filepath', [data_path,'/preprocessed']);
    % Save the EEG structure as a .mat file
    if ~exist(out_path, 'dir')
        mkdir(out_path);
    end
    save(fullfile(out_path, ['preprocessed_', set_name, '.mat']), 'EEG');
end
