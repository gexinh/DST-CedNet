function [sources, info] = dst_sttonnica(sensors, gain, adjacency, components, max_iterations, restarts, seed, sparsity, smoothness, orthogonality, elastic_net)
%DST_STTONNICA Adapter to an externally installed STTONNICA implementation.
%   sensors[C,T], gain[C,V], adjacency[V,V] -> signed sources[V,T].
%   Put STTONNICA.m, KSMG.m and stand.m from the same external package on
%   the MATLAB path before calling. No third-party algorithm is bundled.
%
%   Method: Valdes-Sosa PA, Vega-Hernandez M, Sanchez-Bornot JM,
%   Martinez-Montes E, Bobes MA. Human Brain Mapping 30(6):1898-1910,
%   2009. https://doi.org/10.1002/hbm.20784
%   The inspected external files credit Cuban Neuroscience Center,
%   2009-2016, and Joint China-Cuba Lab, 2016. Their rights and terms
%   remain separate from this newly written adapter's MIT license.
%
%   The source wrapper used four factors, 300 iterations, 25 restarts,
%   sparsity logspace(-8,-4,20), smoothness logspace(1,7,10), mu=1,
%   and elastic-net penalty 1e-8. These are defaults, not a recovered
%   per-experiment optimum. BIC selects penalties/restart at fixed k.
%   The legacy implementation has a squeeze bug at k=1; k must be >=2.

if nargin < 4 || isempty(components), components = 4; end
if nargin < 5 || isempty(max_iterations), max_iterations = 300; end
if nargin < 6 || isempty(restarts), restarts = 25; end
if nargin < 7 || isempty(seed), seed = 0; end
if nargin < 8 || isempty(sparsity), sparsity = logspace(-8, -4, 20); end
if nargin < 9 || isempty(smoothness), smoothness = logspace(1, 7, 10); end
if nargin < 10 || isempty(orthogonality), orthogonality = 1; end
if nargin < 11 || isempty(elastic_net), elastic_net = 1e-8; end

validateattributes(sensors, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'sensors');
validateattributes(gain, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'gain');
[channels, times] = size(sensors);
vertices = size(gain, 2);
if channels < 2 || times < 2 || size(gain, 1) ~= channels
    error('DSTCedNet:STTONNICAShape', 'Require matching channels>=2, time>=2 and gain[C,V].');
end
validateattributes(adjacency, {'numeric', 'logical'}, {'2d', 'real', 'finite', 'size', [vertices, vertices]}, mfilename, 'adjacency');
adjacency = sparse(double(adjacency));
if any(nonzeros(adjacency) < 0) || norm(adjacency - adjacency', 'fro') > 1e-10 * max(1, norm(adjacency, 'fro'))
    error('DSTCedNet:STTONNICAGraph', 'Adjacency must be symmetric and nonnegative.');
end
adjacency = adjacency - spdiags(diag(adjacency), 0, vertices, vertices);
validateattributes(components, {'numeric'}, {'scalar', 'integer', '>=', 2, '<=', min(vertices, times)}, mfilename, 'components');
validateattributes(max_iterations, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'max_iterations');
validateattributes(restarts, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'restarts');
validateattributes(seed, {'numeric'}, {'scalar', 'integer', '>=', 0, '<=', 2^32-1}, mfilename, 'seed');
validateattributes(sparsity, {'numeric'}, {'vector', 'real', 'finite', 'nonempty', 'nonnegative'}, mfilename, 'sparsity');
validateattributes(smoothness, {'numeric'}, {'vector', 'real', 'finite', 'nonempty', 'nonnegative'}, mfilename, 'smoothness');
validateattributes(orthogonality, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, mfilename, 'orthogonality');
validateattributes(elastic_net, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, mfilename, 'elastic_net');

algorithm = which('STTONNICA');
if isempty(algorithm)
    error('DSTCedNet:MissingSTTONNICA', ['STTONNICA.m is an external dependency. ', ...
        'Add a legitimately obtained STTONNICA package with STTONNICA.m, KSMG.m and stand.m to the MATLAB path.']);
end
external_dir = fileparts(algorithm);
for dependency = {'KSMG.m', 'stand.m'}
    if exist(fullfile(external_dir, dependency{1}), 'file') ~= 2
        error('DSTCedNet:MissingSTTONNICA', 'The same external directory must contain %s.', dependency{1});
    end
end

old_path = path;
old_rng = rng;
cleanup = onCleanup(@() restore_context(old_path, old_rng)); %#ok<NASGU>
addpath(external_dir, '-begin');
rng(double(seed), 'twister');
laplacian = spdiags(sum(adjacency, 2), 0, vertices, vertices) - adjacency;
penalty = struct('lmbd', double(sparsity(:)'), 'lmbdsm', double(smoothness(:)'), ...
    'mu', double(orthogonality), 'lmbdenet', double(elastic_net));
options = struct('Nit', double(max_iterations), 'restarts', double(restarts), ...
    'plotresults', 0, 'k', double(components));

% Capture upstream progress output without changing its computation.
evalc('[spatial, temporal, best_indices, degrees_of_freedom, rss, bic, ~, best_bic] = STTONNICA(double(sensors), double(gain), laplacian, penalty, options);');
if ~isequal(size(spatial), [vertices, components]) || ~isequal(size(temporal), [components, times])
    error('DSTCedNet:STTONNICAOutputShape', 'External STTONNICA returned incompatible component shapes.');
end
sources = spatial * temporal;
if ~isreal(sources) || any(~isfinite(sources(:))) || ~isscalar(best_bic) || ~isfinite(best_bic)
    error('DSTCedNet:STTONNICANonfinite', 'External STTONNICA failed to produce a finite real solution and BIC.');
end
if numel(best_indices) ~= 2 || any(best_indices < 1) || ...
        best_indices(1) > numel(sparsity) || best_indices(2) > numel(smoothness)
    error('DSTCedNet:STTONNICASelection', 'External STTONNICA returned invalid penalty-grid indices.');
end
info = struct('method', 'STTONNICA', 'components', components, 'seed', seed, ...
    'max_iterations', max_iterations, 'restarts', restarts, 'best_bic', best_bic, ...
    'best_sparsity', sparsity(best_indices(1)), 'best_smoothness', smoothness(best_indices(2)), ...
    'best_indices', best_indices, 'bic', bic, 'rss', rss, 'degrees_of_freedom', degrees_of_freedom, ...
    'spatial_components', spatial, 'temporal_components', temporal, ...
    'laplacian', 'combinatorial graph D-A', 'reference_transform', 'none', ...
    'output', 'signed spatial_components * temporal_components', ...
    'external_dir', external_dir, 'citation_doi', '10.1002/hbm.20784');
end

function restore_context(old_path, old_rng)
path(old_path);
rng(old_rng);
end
