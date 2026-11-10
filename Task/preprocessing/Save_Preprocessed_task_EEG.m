function Save_Preprocessed_task_EEG(data_path, set_name, out_ipsi, out_contra)
    % Load the cleaned .set file
    in_path = [data_path, '/preprocessed/new'];
    EEG = pop_loadset('filename', ['preprocessed_', set_name, '.set'], 'filepath', in_path);

    disp('Unique event types in EEG:');
    disp(unique({EEG.event.type}));

    if ~exist(out_ipsi, 'dir')
        mkdir(out_ipsi);
    end
    if ~exist(out_contra, 'dir')
        mkdir(out_contra);
    end

    % Split ipsi and contra trials into separate files
    EEG_ipsi = pop_selectevent(EEG, 'type', 'ipsi');
    if ~isempty(EEG_ipsi.event)
        save(fullfile(out_ipsi, ['preprocessed_', set_name, '_ipsi.mat']), 'EEG_ipsi');
        pop_saveset(EEG_ipsi, 'filename', ['preprocessed_', set_name, '_ipsi.set'], 'filepath', out_ipsi);
    else
        warning('No ipsilesional events found, skipping ipsilesional save.');
    end

    EEG_contra = pop_selectevent(EEG, 'type', 'contra');
    if ~isempty(EEG_contra.event)
        save(fullfile(out_contra, ['preprocessed_', set_name, '_contra.mat']), 'EEG_contra');
        pop_saveset(EEG_contra, 'filename', ['preprocessed_', set_name, '_contra.set'], 'filepath', out_contra);
    else
        warning('No contralesional events found, skipping contralesional save.');
    end
end
