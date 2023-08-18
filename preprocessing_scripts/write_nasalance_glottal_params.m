% Write .mat files with nasalance and glottal parameters 

clear all;
close all;

%% Load data

data_dir=dir('data_files_txt/*'); 

for i=3:numel(data_dir)
    subj_path = fullfile('data_files_txt',data_dir(i).name);
    subj_splits = strsplit(subj_path, '/');
    subj_name = cell2mat(subj_splits(2));
    subject_dir = dir(subj_path);
    if subj_name == 'subject8' | subj_name == 'subject9'
        for j=3:numel(subject_dir)
            file_path = fullfile(subj_path,subject_dir(j).name);

            file_splits = strsplit(subject_dir(j).name, '.');
            file_name = cell2mat(file_splits(1));
            All_data = importdata(file_path);

            path_scale_nasal  = fullfile('scale_files', subj_name, strcat(file_name, ' WAV Scale values'),'nasal_mic_sacle.dat');
            nasal_scale = importdata(path_scale_nasal);

            path_scale_oral   = fullfile('scale_files', subj_name, strcat(file_name, ' WAV Scale values'),'oral_mic_sacle.dat');
            oral_scale = importdata(path_scale_oral);

%             path_timestamps   = fullfile('Index_dir', strcat(file_name, '_combined_audio_indices.mat'));
%             indices = load(path_timestamps);
%             timestamps = indices.timestamps;
            
            % set new index dir
            path_timestamps   = fullfile('Index_dir_new_subs', strcat(file_name, '_combined_audio_indices.mat'));
            indices = load(path_timestamps);
            timestamps = indices.timestamps;



    %         combined_signal = nasal_filtered + oral_filtered;

    %         combined_norm = combined_signal/max(abs(combined_signal));

    %         audio_path = fullfile('audio_files/combined_audio',subj_name, strcat(file_name, '_combined_audio.wav'));
    %         audiowrite(audio_path, combined_norm, F_n);

            nasal_parameter = compute_nasalance(All_data, nasal_scale, oral_scale, timestamps);

            egg_parameter = compute_egg_parameter(All_data, subj_name, timestamps);

            param_len = length(nasal_parameter);

            combined_tvs = zeros(2, param_len);

            combined_tvs(1,:) = nasal_parameter(1,:);
            combined_tvs(2,:) = egg_parameter(1,:);

            % write files
            tv_path = fullfile('Nasal_glottal_tvs_trimmed', strcat(file_name, '_nasal_glot.mat'));
            save(tv_path, 'combined_tvs');
       
    
        end
    end
    
   
    
end