% Perform statisitical analysis on correlations computed between the
% Naslance parameter and the HSV intensity for all samples from subject 6

clear all;
close all;
%% Load data

data_dir=dir('data_files_txt/subject2/*.txt'); 

for i=1:numel(data_dir)
    file_path = fullfile('data_files_txt/subject2',data_dir(i).name); 
    file_splits = strsplit(data_dir(i).name, '.');
    file_name = cell2mat(file_splits(1));
    All_data = importdata(file_path);
    
    path_scale_nasal  = fullfile('scale_files/subject2', strcat(file_name, ' WAV Scale values'),'nasal_mic_sacle.dat');
    nasal_scale = importdata(path_scale_nasal);
    
    path_scale_oral   = fullfile('scale_files/subject2', strcat(file_name, ' WAV Scale values'),'oral_mic_sacle.dat');
    oral_scale = importdata(path_scale_oral);
    
    path_data  = fullfile('HSN_matfiles/subject2',strcat("Intensity ", file_name, " data.mat"));
    HSN_data = load(path_data);

%     HSN_i = HSN_data.Intensity;
    HSN_fs = HSN_data.Fs;
    
    nasal_parameter = compute_nasalance(All_data, nasal_scale, oral_scale);
    hsn_intensity = extract_hsn(HSN_data);
    
    [R,p] = compute_correlation(hsn_intensity,nasal_parameter, HSN_fs);
    disp(R(1,2));
    disp(p(1,2));
    
end
