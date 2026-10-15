function [S, info] = dst_loreta(sensors, gain, adjacency, regularization)
%DST_LORETA Random-walk-Laplacian smooth minimum-norm inverse.
% One trial: sensors [channels,time], gain [channels,vertices], adjacency
% [vertices,vertices]. Signed output S [vertices,time].
% M=I-D^(-1)A; C=(M'*M+1e-4*trace(M'*M)/V*I)^(-1), normalized so that
% trace(L*C*L')=channels. With alpha=regularization>0, S=C*L'/(L*C*L'+alpha I)*X.
% With regularization=0, estimate a scalar gamma by evidence iteration for
% covariance I+gamma*L*C*L', then use that same gamma and C for the posterior.
% This maintains the smoothness prior of Gexin Huang's LORETA_utimate.m
% (2020-06-26), fixes its per-batch return, and makes the fitted/prior scales
% consistent. Its original final line used inverse-gamma and an unnormalized
% C although gamma was estimated with normalized C. The fixed-alpha and
% evidence options here are explicit maintained variants, not sLORETA/eLORETA
% or a reproduction of unspecified paper baseline tuning.
% Method reference: Pascual-Marqui, Michel & Lehmann, Int J Psychophysiol
% 18(1),49-65 (1994), doi:10.1016/0167-8760(84)90014-X.
% Copyright (c) 2026 DST-CedNet contributors. MIT License.
[X, L] = dst_inverse_inputs(sensors, gain);
validateattributes(regularization, {'numeric'}, {'scalar','real','nonnegative','finite'});
validateattributes(adjacency, {'numeric','logical'}, {'2d','real','finite','nonnegative'});
n_vertices = size(L, 2);
n_channels = size(L, 1);
if ~isequal(size(adjacency), [n_vertices, n_vertices])
    error('DSTCedNet:AdjacencyShape', 'Adjacency dimensions must match lead-field columns.');
end
A = sparse(double(adjacency));
A = max(A, A');
A = A - spdiags(diag(A), 0, n_vertices, n_vertices);
degrees = full(sum(A, 2));
if any(degrees == 0)
    error('DSTCedNet:IsolatedVertex', 'LORETA requires a neighbor for every vertex.');
end
% Retain only the most recent geometry/lead field. A condition sweep reuses
% this expensive source prior, while every trial still fits its own gamma.
persistent cached
prior_cache_hit = ~isempty(cached) && isequal(cached.gain, L) && isequal(cached.adjacency, A);
if prior_cache_hit
    prior = cached.prior;
    sensor_prior = cached.sensor_prior;
    ridge = cached.ridge;
    normalizer = cached.normalizer;
else
    M = speye(n_vertices) - spdiags(1 ./ degrees, 0, n_vertices, n_vertices) * A;
    precision = M' * M;
    ridge = 1e-4 * full(trace(precision)) / n_vertices;
    prior = full((precision + ridge * speye(n_vertices)) \ speye(n_vertices));
    prior = (prior + prior') / 2;
    sensor_prior = L * prior * L';
    normalizer = n_channels / trace(sensor_prior);
    if ~isfinite(normalizer) || normalizer <= 0
        error('DSTCedNet:PriorScale', 'Cannot normalize the projected source prior.');
    end
    prior = prior * normalizer;
    sensor_prior = sensor_prior * normalizer;
    sensor_prior = (sensor_prior + sensor_prior') / 2;
    cached = struct('gain', L, 'adjacency', A, 'prior', prior, ...
        'sensor_prior', sensor_prior, 'ridge', ridge, 'normalizer', normalizer, ...
        'fixed_alpha', [], 'fixed_operator', []);
end
operator_cache_hit = false;
iterations = 0;
gamma = 0;
if regularization > 0
    operator_cache_hit = ~isempty(cached.fixed_operator) && isequal(cached.fixed_alpha, double(regularization));
    if ~operator_cache_hit
        cached.fixed_operator = (prior * L') / (sensor_prior + double(regularization) * eye(n_channels));
        cached.fixed_alpha = double(regularization);
    end
    S = cached.fixed_operator * X;
    mode = 'fixed_alpha';
elseif ~any(X(:))
    S = zeros(n_vertices, size(X, 2));
    mode = 'evidence';
else
    gamma = 1;
    old_cost = inf;
    for iterations = 1:100
        covariance = eye(n_channels) + gamma * sensor_prior;
        solved = covariance \ X;
        numerator = sum(sum(solved .* (sensor_prior * solved)));
        denominator = size(X, 2) * trace(covariance \ sensor_prior);
        next_gamma = max(gamma * max(real(numerator), 0) / max(real(denominator), realmin), realmin);
        factor = chol(covariance);
        cost = 0.5 * (size(X, 2) * 2 * sum(log(diag(factor))) + sum(sum(X .* solved)));
        relative_change = abs(cost - old_cost) / max(abs(cost), eps);
        gamma = next_gamma;
        if ~isfinite(gamma)
            error('DSTCedNet:EvidenceNumerics', 'Evidence iteration produced a non-finite prior scale.');
        end
        if relative_change < 1e-5
            break;
        end
        old_cost = cost;
    end
    S = gamma * prior * L' * ((eye(n_channels) + gamma * sensor_prior) \ X);
    mode = 'evidence';
end
if ~all(isfinite(S(:)))
    error('DSTCedNet:LoretaNumerics', 'LORETA produced non-finite sources.');
end
info = struct('method', 'loreta', 'mode', mode, 'regularization', double(regularization), ...
    'iterations', iterations, 'gamma', gamma, 'laplacian_ridge', ridge, ...
    'prior_normalizer', normalizer, 'signed_output', true, ...
    'prior_cache_hit', prior_cache_hit, 'operator_cache_hit', operator_cache_hit);
end
