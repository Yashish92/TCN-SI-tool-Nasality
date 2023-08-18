function [R, p] = compute_correlation(hsn_intensity,nasal_parameter, HSN_fs)

hsn_normalized = hsn_intensity;
N = nasal_parameter;
% nasal_sr = 100;

x_rate = length(hsn_normalized)/(HSN_fs*length(N));
xq_rate = length(hsn_normalized)/(HSN_fs*length(hsn_normalized));

x = 0:x_rate:((length(hsn_normalized)/HSN_fs)-x_rate);
xq = 0:xq_rate:((length(hsn_normalized)/HSN_fs)-xq_rate);
N_new = interp1(x,N,xq);
N_new = N_new.';

% R = corrcoef(N_new, hsn_normalized, 'rows', 'complete');

[R,p] = corrcoef(N_new, hsn_normalized, 'rows', 'complete');


end