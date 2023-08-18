function egg_parameter = compute_egg_parameter(All_data, subj_name, timestamps)
% generate glottal TV from EGG data

% [up,lo] =envelope(egg_filt,100,'analytic');

% [up,lo] =envelope(egg_filt(range),1000,'rms');

if subj_name == 'subject4'
    EGG_sig = All_data.data(:,3);
else
    EGG_sig = All_data.data(:,4);
end

% trim the parameters to match trimmed audio
EGG_sig = EGG_sig(timestamps(1)+1:timestamps(2));

F_n = 51200;

% Notch type highpass filter
b = [1 -1];
a = [1 -0.99];

egg_filtered = filter(b,a, EGG_sig);
egg_lowpass = lowpass(egg_filtered, 250, F_n);
% freqz(egg_lowpass);

% figure();
% plot(egg_lowpass);
% hold on;
% plot(egg_filtered, 'r');

% [up,lo] =envelope(egg_lowpass,200,'peak');
% 
% figure();
% plot(egg_filt(range));
% hold on;
% plot(up);
% legend('EGG signal', 'Envelope');

% writematrix(up,'egg_envelope_MR03.txt');

% figure();
% plot(egg_orig.t3(8000:16000));

% compute Hilbert envelope
% env = hilbert_om(egg_filt');
% 
% % butterworth filter for smoothing
% [b,a] = butter(3, 0.001, 'low');
% % [b,a] = cheby1(6,10,0.01);
% env_filt = filtfilt(b,a, env);

% Moving average for smoothing
% windowWidth = 2000;
% env_filt = movmean(env, windowWidth);

% up_filt = movmean(up, windowWidth);

% compute Hilbert envelope
env = abs(hilbert(egg_lowpass));

% dowansample and smooth the signal (using custom developed functions!!)
egg_dsample = condense(env,512);
egg_smooth = fastsmooth(egg_dsample,10,3,1);

% egg_lpass_dsample = condense(egg_lowpass,512);

% nasal_smooth = fastsmooth(egg_dsample,10,3,1);

% Normalize signal to -1, 1 range
min_egg = min(egg_smooth(:));
max_egg = max(egg_smooth(:));
egg_norm1 = (egg_smooth - min_egg)/(max_egg-min_egg);

egg_normalized = egg_norm1*2 -1;

% figure();
% plot(egg_normalized);
% 
% figure();
% plot(egg_smooth);

egg_parameter = egg_normalized;


end