function [TBF, variance_fraction] = dst_extract_basis(measurements_file, input_key, output_file, retained_variance, first_time, last_time, center_channels)
%DST_EXTRACT_BASIS Extract temporal bases with base-MATLAB SVD.
% Input MAT key is a finite real [sensors,time] matrix. first_time/last_time
% are inclusive MATLAB indices; last_time=0 selects the final input sample.
% Optional per-channel temporal mean removal occurs after window selection.
% This is the paper's SVD option, not its real-data SEFA preprocessing.
% Saves TBF [basis_functions,selected_time] and extraction metadata to MAT.

validateattributes(retained_variance, {'numeric'}, {'scalar','positive','finite','<=',1});
validateattributes(first_time, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(last_time, {'numeric'}, {'scalar','integer','nonnegative','finite'});
validateattributes(center_channels, {'numeric','logical'}, {'scalar'});
loaded = load(measurements_file, input_key);
if ~isfield(loaded, input_key)
    error('DSTCedNet:MissingMeasurements', 'Input file has no variable %s.', input_key);
end
X = loaded.(input_key);
validateattributes(X, {'numeric'}, {'2d','real','finite','nonempty'});
X = double(X);
input_size = size(X);
if last_time == 0
    last_time = size(X, 2);
end
if first_time > last_time || last_time > size(X, 2) || last_time - first_time < 1
    error('DSTCedNet:InvalidWindow', 'Select at least two valid time samples.');
end
X = X(:, first_time:last_time);
if center_channels
    X = X - mean(X, 2);
end
[~, singular_matrix, temporal_vectors] = svd(X, 'econ');
energy = diag(singular_matrix).^2;
if isempty(energy) || sum(energy) <= 0
    error('DSTCedNet:ZeroMeasurements', 'Selected measurements have no nonzero signal variance.');
end
cumulative = cumsum(energy) / sum(energy);
n_basis = find(cumulative >= retained_variance - eps, 1, 'first');
TBF = temporal_vectors(:, 1:n_basis)';
normalization_scale = max(abs(TBF(:)));
TBF = TBF / normalization_scale;
variance_fraction = cumulative(n_basis);
method = 'svd';
coordinate_layout = 'basis_functions_by_time';
if ~isempty(output_file)
    save(output_file, 'TBF', 'method', 'retained_variance', 'variance_fraction', ...
        'center_channels', 'first_time', 'last_time', 'input_size', ...
        'normalization_scale', 'coordinate_layout');
end
end
