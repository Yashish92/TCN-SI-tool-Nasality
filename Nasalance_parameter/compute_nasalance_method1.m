clear all;
close all;

%% Load data

% load an individual data recording from a given subject 
path_data  = fullfile('data_files_txt/subject3/Subject 3_01.txt');
All_data = importdata(path_data);

% scale the oral and nasal mic signals with corresponding scale files
path_scale_nasal  = fullfile('scale_files/subject3/Subject 3_01 WAV Scale values/nasal_mic_sacle.dat');
nasal_scale = importdata(path_scale_nasal);

path_scale_oral  = fullfile('scale_files/subject3/Subject 3_01 WAV Scale values/oral_mic_sacle.dat');
oral_scale = importdata(path_scale_oral);

far_scale = 1;

%% Process data

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

nasalance(isnan(nasalance)) = 0;

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


%% generate plots

% % load .mat files
% EGG_sig = load('JM01_EGG.mat');
% EGG_env = load('JM01_egg_envelope.mat');
% F0_sig = load('JM01_f0_new.mat');
% 
% egg_sig = egg_filtered;
% egg_env = up_filt;
% % f0_sig  = F0_sig.f_plot;

% TV_sr =100;
sr = F_n;

%set plot_seg=1 to plot smaller segments
plot_seg =0;

if plot_seg == 1
    t1 = 18.1;
    t2 = 19.8;
    time_frame1 = round(t1*F_n:t2*F_n);
%     N_frame = round(t1*TV_sr:t2*TV_sr);
%     hsn_frame = round(t1*HSN_fs:t2*HSN_fs);
    
    % N = nasalance_new;
    N = nasal_normalized(:,N_frame); % to use the raw nasalance signal
    nasal = nasal_filtered(time_frame1,:);
    oral = oral_filtered(time_frame1,:);
    % waveform = far_filtered;
    waveform = combined_signal(time_frame1,:);
%     hsn_i = hsn_normalized(hsn_frame,:);
    
    % if use LA tv
%     LA_tv_fr = LA_tv(:,N_frame);

    t = linspace(0,length(An)/sr,length(An))';
    t = t(time_frame1);
%     t_hsn = linspace(0,length(An)/sr, length(HSN_i));
%     t_hsn = t_hsn(:,hsn_frame);
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
%     hsn_i = hsn_normalized;

    t = linspace(0,length(An)/sr,length(An))';
%     t = t(time_frame1);
%     t_hsn = linspace(0,length(An)/sr, length(HSN_i));
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
ah = subplots(5,[],[]);
% ah(2).Position(2) = ah(1).Position(2);
% ah(3).Position(4) = ah(3).Position(4)*2;
% ah(1) = [];
c = get(ah(1),'colororder');
sgtitle("Nasalance Plots");
axes(ah(1)); p1(t,waveform); ylabel('Waveform'); ylim([-50,50]);
sgram({s,sr},[],8000,[],ah(2));
axes(ah(3)); p1(t,oral,'color',c(5,:)); ylabel('Oral signal'); ylim([-50,50]);
axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal'); ylim([-50,50]);
% axes(ah(3)); p1(t,oral,'color',c(1,:)); ylabel('Oral signal')
% axes(ah(4)); p1(t,nasal,'color',c(2,:)); ylabel('Nasal signal')
axes(ah(5)); p1(t_nasal,N,'color',c(1,:)); ylabel('Nasalance')

% axes(ah(6)); p1(t,egg_lowpass,'color',[0 .6 0]); ylabel('EGG signal')
% axes(ah(6)); p1(t_hsn,hsn_i,'color',c(3,:)); ylabel('HSN');
% axes(ah(7)); p1(t_nasal,LA_tv_fr,'color',c(4,:)); ylabel('LA TV');
% axes(ah(8)); scatter(t,f0_sig(133120:225280), 2); ylabel('F0')

% line(t([1 end]),.5*[1;1],'color','k','linestyle',':')
set(ah(1:4),'xticklabel',[]);
% set(ah(5),'ytick',[],'xlim',t([1 end])); 
% axes(ah(7));
xlabel('time (secs)')
