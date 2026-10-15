function [S, info] = dst_sparse(sensors, gain, regularization, max_iterations, method, p)
%DST_SPARSE Shared-variance MMV SBL or reweighted FOCUSS inverse.
% One trial: sensors [channels,time], gain [channels,vertices], output signed
% S [vertices,time]. regularization is the direct alpha in alpha*I + L*G*L'.
% method='sbl': fast shared-variance update gamma=sqrt(mean(S.^2)/leverage).
% method='focuss': gamma=mean(S.^2)^(1-p/2), with 0<p<=2 (default p=1).
% These update choices match the original SBL.m flag1=1 and FOCUSS.m [3,1].
% The original sparse_learning.m credits David Wipf; this implementation
% expresses the covariance equations independently and uses no vendor helper.
% It does not implement noise-learning Champagne or temporal covariance SBL.
% References: Wipf & Nagarajan, NeuroImage 44(3),947-966 (2009),
% doi:10.1016/j.neuroimage.2008.02.059; Gorodnitsky, George & Rao,
% Electroencephalogr Clin Neurophysiol 95(4),231-251 (1995),
% doi:10.1016/0013-4694(95)00107-A.
% Copyright (c) 2026 DST-CedNet contributors. MIT License.
if nargin < 6, p = 1; end
[X, L] = dst_inverse_inputs(sensors, gain);
validateattributes(regularization, {'numeric'}, {'scalar','real','positive','finite'});
validateattributes(max_iterations, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(p, {'numeric'}, {'scalar','real','positive','finite','<=',2});
method = lower(char(method));
if ~strcmp(method, 'sbl') && ~strcmp(method, 'focuss')
    error('DSTCedNet:SparseMethod', 'method must be sbl or focuss.');
end
n_vertices = size(L, 2);
n_times = size(X, 2);
gamma = double(sum(L.^2, 1)' > 0);
S = zeros(n_vertices, n_times);
change = 0;
iteration = 0;
minimum_gamma = 1e-16;
minimum_change = 1e-12;
if any(X(:))
    for iteration = 1:double(max_iterations)
        active = find(gamma > minimum_gamma);
        if isempty(active)
            S(:) = 0;
            break;
        end
        local_gain = L(:, active);
        weighted_gain = local_gain .* gamma(active)';
        covariance = weighted_gain * local_gain' + double(regularization) * eye(size(L, 1));
        solved_signal = covariance \ X;
        local_sources = weighted_gain' * solved_signal;
        source_power = sum(local_sources.^2, 2) / n_times;
        next_gamma = zeros(n_vertices, 1);
        if strcmp(method, 'sbl')
            % diag(L' C^-1 L); cancellation of the old gamma avoids a 0/0.
            leverage = sum(local_gain .* (covariance \ local_gain), 1)';
            next_gamma(active) = sqrt(max(source_power, 0) ./ max(leverage, realmin));
        else
            next_gamma(active) = max(source_power, 0).^(1 - double(p)/2);
        end
        previous = S;
        S(:) = 0;
        S(active, :) = local_sources;
        change = max(abs(S(:) - previous(:)));
        gamma = next_gamma;
        if ~all(isfinite(S(:))) || ~all(isfinite(gamma))
            error('DSTCedNet:SparseNumerics', 'The sparse inverse produced non-finite values.');
        end
        if change < minimum_change
            break;
        end
    end
else
    gamma(:) = 0;
end
info = struct('method', method, 'regularization', double(regularization), ...
    'iterations', iteration, 'max_iterations', double(max_iterations), ...
    'source_change', change, 'gamma', gamma, 'p', double(p), ...
    'prune_threshold', minimum_gamma, 'stopping_threshold', minimum_change, ...
    'signed_output', true);
end
