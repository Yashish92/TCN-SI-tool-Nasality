function nasal_parameter = compute_nasalance(All_data, nasal_scale, oral_scale, timestamps)

% computes a normalized nasalance parameter (-1,1) from oral and nasal mic
% signls

far_scale = 1;
nasal_sig = All_data.data(:,1)*nasal_scale;
oral_sig = All_data.data(:,2)*oral_scale;
far_sig = All_data.data(:,3)*far_scale;

nasal_sig = nasal_sig(timestamps(1)+1:timestamps(2));
oral_sig = oral_sig(timestamps(1)+1:timestamps(2));
far_sig = far_sig(timestamps(1)+1:timestamps(2));
% EGG_sig = All_data.data(:,4);

F_n = 51200;

% Notch type highpass filter
b = [1 -1];
a = [1 -0.99];

nasal_filtered = filter(b,a, nasal_sig);
oral_filtered = filter(b,a, oral_sig);
far_filtered = filter(b,a, far_sig);
% egg_filtered = filter(b,a, EGG_sig);

combined_signal = nasal_filtered + oral_filtered;

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

% dowansample and smooth the signal (using custom developed functions!!)
nasal_dsample = condense(nasalance,512);
nasal_smooth = fastsmooth(nasal_dsample,10,3,1);

% Normalize signal to -1, 1 range
min_nasal = min(nasal_smooth(:));
max_nasal = max(nasal_smooth(:));
nasal_norm1 = (nasal_smooth - min_nasal)/(max_nasal-min_nasal);

nasal_normalized = nasal_norm1*2 -1;
nasal_parameter = nasal_normalized;


end