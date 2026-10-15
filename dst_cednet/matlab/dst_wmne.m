function [S, info] = dst_wmne(sensors, gain, regularization, depth)
%DST_WMNE Weighted minimum norm with the legacy column-norm weighting.
% X [channels,time], L [channels,vertices] -> signed S [vertices,time].
% D_j = ||L(:,j)||_2^(-depth), S = D L' (L D L' + alpha I)^(-1) X.
% regularization is alpha > 0, without trace scaling. The original
% Results/Algorithms/wMNE.m used lambda D L' (I+lambda L D L')^(-1),
% so alpha=1/lambda; depth=1 reproduces its weighting. Its final abs()
% was intended for plots; signed time courses are retained here.
% Method reference: Dale and Sereno, J Cogn Neurosci 5(2), 162-176 (1993),
% doi:10.1162/jocn.1993.5.2.162. Weighting details above define this variant.
% Copyright (c) 2026 DST-CedNet contributors. MIT License.
if nargin < 4, depth = 1; end
[X, L] = dst_inverse_inputs(sensors, gain);
validateattributes(regularization, {'numeric'}, {'scalar','real','positive','finite'});
validateattributes(depth, {'numeric'}, {'scalar','real','nonnegative','finite'});
column_norm = sqrt(sum(L.^2, 1));
weights = zeros(size(column_norm));
nonzero = column_norm > 0;
weights(nonzero) = column_norm(nonzero).^(-double(depth));
weighted_gain = L .* weights;
covariance = weighted_gain * L' + double(regularization) * eye(size(L, 1));
S = weighted_gain' * (covariance \ X);
info = struct('method', 'wmne-matlab', 'regularization', double(regularization), ...
    'depth', double(depth), 'iterations', 1, 'signed_output', true);
end
