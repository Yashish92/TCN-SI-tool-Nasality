% Write .wav files by combining oral and nasal mic signals 

clear all;
close all;

%% Load data

data_dir=dir('data_files_txt/*'); 

for i=3:numel(data_dir)
    sub_id = data_dir(i).name;
    % update this every time for new subjects or remove the if case to run
    % on all subjects
    if sub_id == 'subject8' | sub_id == 'subject9' 
        subj_path = fullfile('data_files_txt',data_dir(i).name);
        subj_splits = strsplit(subj_path, '/');
        subj_name = cell2mat(subj_splits(2));
        subject_dir = dir(subj_path);
        for j=3:numel(subject_dir)
            file_path = fullfile(subj_path,subject_dir(j).name);

            file_splits = strsplit(subject_dir(j).name, '.');
            file_name = cell2mat(file_splits(1));
            All_data = importdata(file_path);

            path_scale_nasal  = fullfile('scale_files', subj_name, strcat(file_name, ' WAV Scale values'),'nasal_mic_sacle.dat');
            nasal_scale = importdata(path_scale_nasal);

            path_scale_oral   = fullfile('scale_files', subj_name, strcat(file_name, ' WAV Scale values'),'oral_mic_sacle.dat');
            oral_scale = importdata(path_scale_oral);

            nasal_sig = All_data.data(:,1)*nasal_scale;
            oral_sig = All_data.data(:,2)*oral_scale;
    %         far_sig = All_data.data(:,3)*far_scale;
    %         EGG_sig = All_data.data(:,4);

            F_n = 51200;

            % Notch type highpass filter
            b = [1 -1];
            a = [1 -0.99];

            nasal_filtered = filter(b,a, nasal_sig);
            oral_filtered = filter(b,a, oral_sig);
    %         far_filtered = filter(b,a, far_sig);
    %         egg_filtered = filter(b,a, EGG_sig);

            combined_signal = nasal_filtered + oral_filtered;

            combined_norm = combined_signal/max(abs(combined_signal));

            audio_path = fullfile('audio_files/combined_audio',subj_name, strcat(file_name, '_combined_audio.wav'));
            audiowrite(audio_path, combined_norm, F_n);
        
    
        end
    end
    
   
    
end