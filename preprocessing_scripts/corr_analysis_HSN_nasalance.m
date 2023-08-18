clear all;
close all;

%% Load data

path_data  = fullfile('data_files_txt/subject2/Subject 2_01.txt');
All_data = importdata(path_data);

path_scale_nasal  = fullfile('scale_files/subject2/Subject 2_01 WAV Scale values/nasal_mic_sacle.dat');
nasal_scale = importdata(path_scale_nasal);

path_scale_oral  = fullfile('scale_files/subject2/Subject 2_01 WAV Scale values/oral_mic_sacle.dat');
oral_scale = importdata(path_scale_oral);

far_scale = 1;

%% Load LA tv for plots
TVs = load('Timing_analysis_TVs/Subject 2_01_combined_audio_tv_predict.mat');
all_tvs = TVs.tv;
LA_tv = all_tvs(1,:);

% TVs = load('Timing_analysis_TVs/ganesh_SI/Subject 6_01_combined_audio.mat');
% all_tvs = TVs.TV;
% LA_tv = all_tvs(:,1);
% LA_tv = LA_tv.';


%% Load HSN data if available

path_data  = fullfile('HSN_matfiles/subject2/Intensity Subject 2_01 data.mat');
HSN_data = load(path_data);

HSN_i = HSN_data.Intensity;
HSN_fs = HSN_data.Fs;


%% Process data 

% % MR03 scales
% nasal_scale = 2.997;
% oral_scale = 5.222;

% % Run 03 scales
% nasal_scale = 2.055;
% oral_scale = 3.650;
% far_scale = 0.254;

% % AG_01 scales
% nasal_scale = 3.038;
% oral_scale = 11.334;
% far_scale = 0.671;

% % JM_08 scales
% nasal_scale = 5.586;
% oral_scale = 4.232;
% far_scale = 0.303;

% % JM_01 scales
% nasal_scale = 2.340;
% oral_scale = 6.672;
% far_scale = 1;

% 
% % KM_05 scales
% nasal_scale = 1.625;
% oral_scale = 2.864;
% far_scale = 0.282;

nasal_sig = All_data.data(:,1)*nasal_scale;
oral_sig = All_data.data(:,2)*oral_scale;
far_sig = All_data.data(:,3)*far_scale;
EGG_sig = All_data.data(:,4);

% plot(EGG_sig)

F_n = 51200;

% Notch type highpass filter
b = [1 -1];
a = [1 -0.99];

nasal_filtered = filter(b,a, nasal_sig);
oral_filtered = filter(b,a, oral_sig);
far_filtered = filter(b,a, far_sig);
egg_filtered = filter(b,a, EGG_sig);

combined_signal = nasal_filtered + oral_filtered;

% Smooth HSN signal
HSN_smoothed = fastsmooth(HSN_i,50,3,1);

% Normalize HSN_i signal to -1, 1 range
min_hsn = min(HSN_smoothed(:));
max_hsn = max(HSN_smoothed(:));
hsn_norm1 = (HSN_smoothed - min_hsn)/(max_hsn-min_hsn);

hsn_normalized = hsn_norm1*2 -1;

figure();
plot(hsn_normalized);
% hold on;
% plot(HSN_smoothed);

% plot(combined_signal);
% hold on;
% plot(oral_filtered);
% hold on;
% plot(nasal_filtered);
% legend('combined', 'oral', 'nasal');

% extract part of the utterance (2.9s to 5.0s)

% nasal_filtered = nasal_filtered(74240:128000);
% oral_filtered = oral_filtered(74240:128000);
% far_filtered = far_filtered(74240:128000);

% % Run voice activity detector
% idx = detectSpeech(oral_filtered,F_n);
% 
% % detectSpeech(oral_filtered,F_n);

% generate new oral and nasal signals with silences made zero
% buffer = 7000 ;% 7000; 
% nasal_filt_new = zeros(length(nasal_filtered),1);
% oral_filt_new = zeros(length(oral_filtered),1);
% 
% for i=1:size(idx,1)
%     nasal_filt_new(idx(i,1)-buffer:idx(i,2)+buffer) = nasal_filtered(idx(i,1)-buffer:idx(i,2)+buffer);
%     oral_filt_new(idx(i,1)-buffer:idx(i,2)+buffer) = oral_filtered(idx(i,1)-buffer:idx(i,2)+buffer);
% end
% 
% 
% figure();
% plot(oral_filt_new);
% hold on;
% figure();
% plot(oral_filtered);
% figure;
% plot(nasal_filt_new);
% hold on;
% figure;
% plot(nasal_filtered);
% legend('original', 'filtered');
% % 
% % % plot(abs(fftshift(fft(nasal_sig))));
% % 
% % [eeg_mocha,F_mocha] = audioread('laryngograph.wav');
% % 
% % figure();
% % plot(eeg_mocha);
% % title('laryngograph from MOCHA-TIMIT');
% 
% T_s = 1/F_n ;
% sig_len = length(nasal_sig)/F_n;
% time = 0:T_s:(length(nasal_sig)/F_n)-T_s ;
% 
% % 
% % 
% % 
% % figure;
% % plot(time,nasal_filtered);
% % title('Nasal signal')
% % figure;
% % plot(time,oral_filtered);
% % title('Oral signal')
% % figure;
% % plot(time,far_filtered);
% % title('Far field signal')
% 
% % egg_hilbert = hilbert(EGG_sig);
% % env = abs(egg_hilbert);
% % % 
% % figure();
% % plot(EGG_sig);
% % title('Our EGG data');
% % hold on;
% % plot(env);
% 
% 
% % title('EGG signal')
% % hold on;
% % plot(egg_filtered);
% % write first 10 sec to a .wav file
% 
% % nasal_param = abs(nasal_filtered./(nasal_filtered + oral_filtered));
% % nasal_param = nasal_filtered.^2./oral_filtered.^2;
% % 
% % 
% % 
% % figure;
% % plot(time, nasal_param)
% % title('Nasal Parameter')
% 
% % subplot(nasal_sig, oral_sig)
% % 


%% compute Nasalance parameter using RMS values

% if silence removed used
% oral_filtered = oral_filt_new;
% nasal_filtered = nasal_filt_new;

% Square the signal.
ySquared = oral_filtered .^ 2;
% Get a moving average within a window
windowWidth = 1000; % changed from 1000
meanSquareY = movmean(ySquared, windowWidth);
% Take the square root of the mean square signal.
Ao = sqrt(meanSquareY);

% Square the signal.
ySquared = nasal_filtered .^ 2;
% Get a moving average within a window
% windowWidth = 1000;
meanSquareY = movmean(ySquared, windowWidth);
% Take the square root of the mean square signal.
An = sqrt(meanSquareY);

% % for even window size
% delay = windowWidth/2 ; 
% 
% Ao = Ao(delay:size(Ao));
% An = An(delay:size(An));

% figure;
% plot(Ao);


Ano = An + Ao;
% k = find(Ano > max(Ano)*.05,1);
% nn = mode(An(1:k));
% oo = mode(Ao(1:k));
% offset = 0 - nn/(nn+oo);
% nasalance = An ./ (An + Ao) + offset;

nasalance = An ./ (An + Ao);

nasalance(nasalance == NaN) = 0;

% post filter nasalance raw parameter
% nasal_lowpass = lowpass(nasalance, 0.5, F_n);

nasal_dsample = condense(nasalance,512);
nasal_smooth = fastsmooth(nasal_dsample,10,3,1);

% Normalize signal to -1, 1 range
min_nasal = min(nasal_smooth(:));
max_nasal = max(nasal_smooth(:));
nasal_norm1 = (nasal_smooth - min_nasal)/(max_nasal-min_nasal);

nasal_normalized = nasal_norm1*2 -1;

figure();
plot(nasal_normalized);

figure();
plot(nasalance);



% nasalance = An ./ (An + Ao);

% plot Nasal and Oral RMS signals
% figure;
% plot(An);
% 
% figure;
% plot(nasalance);

% % remove parameter value at areas of silences
% nasalance_new = zeros(length(Ano),1);

% figure;
% ah = subplots(4,[],1);
% ah(2).Position(2) = ah(1).Position(2);
% ah(3).Position(4) = ah(3).Position(4)*2;
% ah(1) = [];
% c = get(ah(1),'colororder');
% axes(ah(1)); p1(t,An); ylabel('RMS (nasal)')
% axes(ah(2)); p1(t,Ao,'color',c(5,:)); ylabel('RMS (oral)')
% axes(ah(3)); p1(t,nasalance,'color',c(2,:)); ylabel('Nasalance (raw)')
% % sgram({s,sr},[],5000,[],ah(4));
% % % axes(ah(3)); p1(t,oral,'color',c(1,:)); ylabel('Oral signal')
% % % axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
% % axes(ah(5)); p1(t,N,'color',c(1,:)); ylabel('Nasalance')
% 
% % axes(ah(6)); p1(t,egg_sig,'color',[0 .6 0]); ylabel('EGG signal')
% % axes(ah(7)); p1(t,egg_env,'color',c(3,:)); ylabel('EGG envelope');ylim([0,0.3])
% % axes(ah(8)); scatter(t,f0_sig, 2); ylabel('F0')
% 
% line(t([1 end]),.7*[1;1],'color','k','linestyle',':')
% set(ah(1:3),'xticklabel',[]);
% set(ah(4),'ytick',[],'xlim',t([1 end])); 
% xlabel('time (secs)')


% buffer = 3000;
% for i=1:size(idx,1)
%     nasalance_new(idx(i,1):idx(i,2)+buffer) = nasalance(idx(i,1):idx(i,2)+buffer);
% end

% % generate the binary Nasal signal
% nasal_ampl_thresh = 0.3; %0.07; % needs to be fine-tuned
% for i=1:length(nasalance_new)
%     if nasalance(i) > nasal_ampl_thresh
%         nasalance_new(i) = nasalance(i);
%     else
%         nasalance_new(i) = 0;
%     end
%     
% end
% 
% t = nasalance_new ~= 0;
% idx_strt = findstr([0 t'], [0 1]);  %gives indices of beginning of groups
% idx_stop = findstr([t' 0], [1 0]);    %gives indices of end of groups
% 
% nasal_dur_thresh = 8000; % needs to be fine-tuned
% for i=1:length(idx_strt)
%     if idx_stop(i) - idx_strt(i) < nasal_dur_thresh
%         nasalance_new(idx_strt(i):idx_stop(i)) = 0;
%     end  
% end

% figure;
% plot(nasalance);
% hold on;
% plot(nasalance_new);
% 
% figure;
% ah = subplots(5,[],1);
% ah(2).Position(2) = ah(1).Position(2);
% ah(3).Position(4) = ah(3).Position(4)*2;
% ah(1) = [];
% c = get(ah(1),'colororder');
% axes(ah(1)); p1(t,oral_filtered,'color',c(4,:)); ylabel('Oral mic signal')
% axes(ah(2)); p1(t,nasal_filtered); ylabel('Nasal mic signal')
% axes(ah(3)); p1(t,nasalance,'color',c(5,:)); ylabel('Nasalance (raw))')
% axes(ah(4)); p1(t,nasalance_new,'color',c(2,:)); ylabel('Nasalance (final))')
% % sgram({s,sr},[],5000,[],ah(4));
% % axes(ah(3)); p1(t,oral,'color',c(1,:)); ylabel('Oral signal')
% % % axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
% % axes(ah(5)); p1(t,N,'color',c(1,:)); ylabel('Nasalance')
% 
% % axes(ah(6)); p1(t,egg_sig,'color',[0 .6 0]); ylabel('EGG signal')
% % axes(ah(7)); p1(t,egg_env,'color',c(3,:)); ylabel('EGG envelope');ylim([0,0.3])
% % axes(ah(8)); scatter(t,f0_sig, 2); ylabel('F0')
% 
% line(t([1 end]),.7*[1;1],'color','k','linestyle',':')
% set(ah(1:4),'xticklabel',[]);
% set(ah(5),'ytick',[],'xlim',t([1 end])); 
% xlabel('time (secs)')

% write to open as a time aligned file in wavesurfer
% writematrix(nasalance_new, 'nasalance_JM01_filtered.txt');
% writematrix(nasalance_new, 'nasalance_run03_sil_rem.txt');

%% generate plots from Dr.Mark functions for proposal

% % load .mat files
% EGG_sig = load('JM01_EGG.mat');
% EGG_env = load('JM01_egg_envelope.mat');
% F0_sig = load('JM01_f0_new.mat');
% 
% egg_sig = egg_filtered;
% egg_env = up_filt;
% % f0_sig  = F0_sig.f_plot;

TV_sr =100;
sr = F_n;

%plot smaller segments
plot_seg =1;
if plot_seg == 1
    t1 = 18.1;
    t2 = 19.8;
    time_frame1 = round(t1*F_n:t2*F_n);
    N_frame = round(t1*TV_sr:t2*TV_sr);
    hsn_frame = round(t1*HSN_fs:t2*HSN_fs);
    
    % N = nasalance_new;
    N = nasal_normalized(:,N_frame); % to use the raw nasalance signal
    nasal = nasal_filtered(time_frame1,:);
    oral = oral_filtered(time_frame1,:);
    % waveform = far_filtered;
    waveform = combined_signal(time_frame1,:);
    hsn_i = hsn_normalized(hsn_frame,:);
    
    % if use LA tv
    LA_tv_fr = LA_tv(:,N_frame);

    t = linspace(0,length(An)/sr,length(An))';
    t = t(time_frame1);
    t_hsn = linspace(0,length(An)/sr, length(HSN_i));
    t_hsn = t_hsn(:,hsn_frame);
    t_nasal = linspace(0,length(An)/sr, length(nasal_smooth));
    t_nasal = t_nasal(N_frame);
    t_nasal = t_nasal - t_nasal(1);
else
    % N = nasalance_new;
    N = nasal_normalized; % to use the raw nasalance signal
    nasal = nasal_filtered;
    oral = oral_filtered;
    % waveform = far_filtered;
    waveform = combined_signal;
    hsn_i = hsn_normalized;

    t = linspace(0,length(An)/sr,length(An))';
%     t = t(time_frame1);
    t_hsn = linspace(0,length(An)/sr, length(HSN_i));
%     t_hsn = t_hsn(:,hsn_frame);
    t_nasal = linspace(0,length(An)/sr, length(nasal_smooth));
%     t_nasal = t_nasal(N_frame);
end

% % N = nasalance_new;
% N = nasal_normalized(:,N_frame); % to use the raw nasalance signal
% nasal = nasal_filtered(time_frame1,:);
% oral = oral_filtered(time_frame1,:);
% % waveform = far_filtered;
% waveform = combined_signal(time_frame1,:);
% hsn_i = hsn_normalized(hsn_frame,:);
% 
% sr = F_n;
% 
% t = linspace(0,length(An)/sr,length(An))';
% t = t(time_frame1);
% t_hsn = linspace(0,length(An)/sr, length(HSN_i));
% t_hsn = t_hsn(:,hsn_frame);
% t_nasal = linspace(0,length(An)/sr, length(nasal_smooth));
% t_nasal = t_nasal(N_frame);
% only for 2 senetence visualization
% t = t(133120:225280);
% s = nasal + oral; s = s ./ max(abs(s));

s= waveform;

% % setting range to extract given range from utterance
% range = 51200*0.5:51200*7.5;
% t = t(range);
% waveform = waveform(range);
% s = s(range);
% oral = oral(range);
% nasal = nasal(range);
% N = N(range);
% egg_lowpass = egg_lowpass(range);
% egg_env = egg_env(range);


% plotting sentence 3 and 4

figure;
ah = subplots(8,[],1);
ah(2).Position(2) = ah(1).Position(2);
ah(3).Position(4) = ah(3).Position(4)*2;
ah(1) = [];
c = get(ah(1),'colororder');
sgtitle("It's hoe me");
axes(ah(1)); p1(t,waveform); ylabel('Waveform'); ylim([-30,40]);
sgram({s,sr},[],8000,[],ah(2));
axes(ah(3)); p1(t,oral,'color',c(5,:)); ylabel('Oral signal'); ylim([-30,40]);
axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal'); ylim([-30,40]);
% axes(ah(3)); p1(t,oral,'color',c(1,:)); ylabel('Oral signal')
% axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
axes(ah(5)); p1(t_nasal,N,'color',c(1,:)); ylabel('Nasalance')

% axes(ah(6)); p1(t,egg_lowpass,'color',[0 .6 0]); ylabel('EGG signal')
axes(ah(6)); p1(t_hsn,hsn_i,'color',c(3,:)); ylabel('HSN');
axes(ah(7)); p1(t_nasal,LA_tv_fr,'color',c(4,:)); ylabel('LA TV');
% axes(ah(8)); scatter(t,f0_sig(133120:225280), 2); ylabel('F0')

line(t([1 end]),.8*[1;1],'color','k','linestyle',':')
set(ah(1:5),'xticklabel',[]);
set(ah(6),'ytick',[],'xlim',t_hsn([1 end])); 
axes(ah(7));
xlabel('time (secs)')

% % find regions where N > 0
% idx = find(N > .0 & An > .2*max(An));
% [h,len] = FindExtents(idx);
% ht = [idx(h)/sr , (idx(h)+len-1)/sr];
% y = [0 1 1 0];


% % second figure
% 
% figure;
% ah = subplots(6,[],1);
% ah(2).Position(2) = ah(1).Position(2);
% ah(3).Position(4) = ah(3).Position(4)*2;
% ah(1) = [];
% c = get(ah(1),'colororder');
% axes(ah(1)); p1(t,waveform(133120:225280)); ylabel('Waveform')
% % axes(ah(2)); p1(t,oral,'color',c(5,:)); ylabel('Oral signal')
% % axes(ah(3)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
% sgram({s(133120:225280),sr},[],5000,[],ah(2));
% % axes(ah(3)); p1(t,oral,'color',c(1,:)); ylabel('Oral signal')
% % axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
% axes(ah(3)); p1(t,N(133120:225280),'color',c(1,:)); ylabel('Nasalance')
% 
% % axes(ah(6)); p1(t,egg_sig,'color',[0 .6 0]); ylabel('EGG signal')
% axes(ah(4)); p1(t,egg_env(133120:225280),'color',c(3,:)); ylabel('EGG envelope');ylim([0,0.3])
% axes(ah(5)); scatter(t,f0_sig(133120:225280), 2); ylabel('F0')
% 
% line(t([1 end]),.7*[1;1],'color','k','linestyle',':')
% set(ah(1:5),'xticklabel',[]);
% set(ah(6),'ytick',[],'xlim',t([1 end])); 
% 
% % % find regions where N > 0
% % idx = find(N > .0 & An > .2*max(An));
% % [h,len] = FindExtents(idx);
% % ht = [idx(h)/sr , (idx(h)+len-1)/sr];
% % y = [0 1 1 0];
% axes(ah(5));
% xlabel('time (secs)')
% % for k = 1 : size(ht,1)
% %     x = [ht(k,1) ht(k,1) ht(k,2) ht(k,2)];
% %     patch(x,y,c(3,:),'FaceAlpha',.1,'EdgeColor','none');
% % end
% 
% % clear nasalance_new
% 
% % write to open as a time aligned file in wavesurfer
% writematrix(nasalance_new, 'nasalance_JM01.txt');
% writematrix(f0_sig', 'F0_JM01.txt');
% writematrix(egg_env, 'EGG_env_JM01.txt');

%% compute time series correlation between HSN and Nasalance parameter

% upsample HSN signal to match the Nasalance signal

% x_rate = length(An)/(sr*length(HSN_i));
% xq_rate = length(An)/(sr*length(An));
% 
% x = 0:x_rate:((length(An)/sr)-x_rate);
% xq = 0:xq_rate:((length(An)/sr)-xq_rate);
% HSN_new = interp1(x,HSN_i,xq);
% HSN_new = HSN_new.';

nasal_sr = 100;
x_rate = length(hsn_normalized)/(HSN_fs*length(N));
xq_rate = length(hsn_normalized)/(HSN_fs*length(hsn_normalized));

x = 0:x_rate:((length(hsn_normalized)/HSN_fs)-x_rate);
xq = 0:xq_rate:((length(hsn_normalized)/HSN_fs)-xq_rate);
N_new = interp1(x,N,xq);
N_new = N_new.';

R = corrcoef(N_new, hsn_normalized, 'rows', 'complete');

%% Write normalized audio files% Normalize signal 
nasal_norm = nasal_filtered/max(abs(nasal_filtered));
oral_norm = oral_filtered/max(abs(oral_filtered));
far_norm = far_filtered/max(abs(far_filtered));

combined_norm = combined_signal/max(abs(combined_signal));
% egg_norm  = egg_filtered/max(abs(egg_filtered));
% egg_orig_norm = EGG_sig/max(abs(EGG_sig));

% audiowrite('nasal_filt_MR03.wav', nasal_norm,F_n);
% audiowrite('oral_filt_MR03.wav', oral_norm,F_n);
% % audiowrite('egg_filt.wav', egg_norm,51200);
% audiowrite('egg_orig_MR03.wav', egg_orig_norm,F_n);

% samples_len = (209920:256000);

% extract 4 seconds of data and plot
% audiowrite('JM01_nasal_2.9s_to_5s.wav', nasal_norm, F_n);
% audiowrite('JM01_oral_2.9s_to_5s.wav', oral_norm, F_n);
% audiowrite('JM01_farfield_2.9s_to_5s.wav', far_norm, F_n);

audiowrite('audio_files/subject6/Subject 6_06 Sound files/combined_farfield.wav', combined_norm, F_n);
% audiowrite('egg_filt.wav', egg_norm,51200);
% audiowrite('egg_orig_MR03.wav', egg_orig_norm,F_n);

 %% Read intensity files, write to .txt files for wavesurfer
% file_nasal = '/home/yashish/Academics/Yashish Personal/Research/Nasal_TVs_project/Data/Intensity_files/run03_nasal_silent_rem.txt';
% file_oral = '/home/yashish/Academics/Yashish Personal/Research/Nasal_TVs_project/Data/Intensity_files/run03_oral_silent_rem.txt';
% % 
% % 
% % [nasal]=textscan(fileID_nasal,'%f');
% % [oral]=textscan(fileID_oral,'%f');
% % 
% % fclose(fileID_oral);
% % fclose(fileID_nasal);
% 
% [nasal, oral, nasal_param]= calc_nasalance(file_nasal, file_oral, idx);
% 
% figure;
% plot(nasal);
% hold on;
% plot(oral);
% legend({'Nasal','Oral'});
% % hold off;
% % nasal_para=(100*nasal)/(nasal+oral);
% figure;
% plot(nasal_param);
% legend('Nasal Parameter');
% 
% 
% writematrix(nasal_param, 'normalized_nasal_param_run03_v1.txt');