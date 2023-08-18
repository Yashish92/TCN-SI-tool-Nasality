function hsn_intensity = extract_hsn(HSN_data)

HSN_i = HSN_data.Intensity;
HSN_fs = HSN_data.Fs;

% Smooth HSN signal
HSN_smoothed = fastsmooth(HSN_i,50,3,1);

% Normalize HSN_i signal to -1, 1 range
min_hsn = min(HSN_smoothed(:));
max_hsn = max(HSN_smoothed(:));
hsn_norm1 = (HSN_smoothed - min_hsn)/(max_hsn-min_hsn);

hsn_normalized = hsn_norm1*2 -1;
hsn_intensity = hsn_normalized;



end