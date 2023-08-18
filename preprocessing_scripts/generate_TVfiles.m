clear all;
close all;

data_nasalTVs       =dir('Nasal_glottal_tvs_trimmed/*.mat'); 
data_3TVs       =dir('APP_tvs_trimmed/*.mat'); 
save_path = fullfile('Final_tvs_trimmed_ext/');

for i=1:numel(data_nasalTVs)
  file_name_data_9TVs = fullfile('Nasal_glottal_tvs_trimmed',data_nasalTVs(i).name);
  file_name_data_3TVs = fullfile('APP_tvs_trimmed',data_3TVs(i).name);
  T_data_9TVs = cell2mat(struct2cell(load(file_name_data_9TVs)));
  T_data_9TVs = T_data_9TVs.';
  struct_file = load(file_name_data_3TVs);  
  T_data_3TVs = struct_file.fin_per_val;
  T_data_3TVs = T_data_3TVs.';
%   data_length_6TVs = length(T_data_6TVs);
  data_length_3TVs = length(T_data_3TVs);
  T_data_9TVs = T_data_9TVs((1:data_length_3TVs),:);
  T_data_3TVs = T_data_3TVs((1:data_length_3TVs),:);
  T_data_full = [T_data_9TVs T_data_3TVs];
  matname = fullfile(save_path, [data_nasalTVs(i).name]);
  save(matname,'T_data_full');
end