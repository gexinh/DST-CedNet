function [W, centers, achieved_cm2, requested_cm2] = dst_sample_patches(cortex_file, n_samples, n_sources, min_area_cm2, max_area_cm2, condition_areas_cm2, seed, allow_overlap)
%DST_SAMPLE_PATCHES Sample contiguous cortical patches using base MATLAB.
% [W, centers, achieved_cm2, requested_cm2] = dst_sample_patches(...)
% Cortex geometry uses meters and 1-based Faces/VertConn indices. W has size
% [n_vertices,n_sources,n_samples]; centers has [n_samples,n_sources] and
% contains MATLAB 1-based vertex indices. Areas are measured in cm^2.
% Explicit condition areas must be a row vector with one entry per source.
% Empty condition areas select independent Uniform(min_area,max_area) draws.
% The function reads only cortex_file, creates no files, and restores rng.

validateattributes(n_samples, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(n_sources, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(min_area_cm2, {'numeric'}, {'scalar','nonnegative','finite'});
validateattributes(max_area_cm2, {'numeric'}, {'scalar','nonnegative','finite','>=',min_area_cm2});
validateattributes(seed, {'numeric'}, {'scalar','integer','nonnegative','finite','<=',2^32-1});
validateattributes(allow_overlap, {'numeric','logical'}, {'scalar'});
if ~isempty(condition_areas_cm2)
    validateattributes(condition_areas_cm2, {'numeric'}, {'vector','nonnegative','finite','numel',n_sources});
    condition_areas_cm2 = reshape(double(condition_areas_cm2), 1, n_sources);
end

loaded = load(cortex_file);
if ~isfield(loaded, 'Cortex')
    error('DSTCedNet:MissingCortex', 'Expected a Cortex structure in %s.', cortex_file);
end
cortex = loaded.Cortex;
if ~isfield(cortex, 'Vertices') || ~isfield(cortex, 'Faces')
    error('DSTCedNet:MissingGeometry', 'Cortex must contain Vertices and Faces.');
end
vertices = double(cortex.Vertices);
faces = double(cortex.Faces);
validateattributes(vertices, {'double'}, {'2d','ncols',3,'finite','nonempty'});
validateattributes(faces, {'double'}, {'2d','ncols',3,'finite','integer','positive','nonempty'});
n_vertices = size(vertices, 1);
if max(faces(:)) > n_vertices
    error('DSTCedNet:InvalidFaces', 'Faces must contain 1-based vertex indices.');
end

% Triangle areas divided equally between their three vertices. No external
% geometry package is needed; coordinates are meters, hence area*1e4 is cm2.
edge_a = vertices(faces(:, 2), :) - vertices(faces(:, 1), :);
edge_b = vertices(faces(:, 3), :) - vertices(faces(:, 1), :);
triangle_cm2 = 0.5 * sqrt(sum(cross(edge_a, edge_b, 2).^2, 2)) * 1e4;
vertex_cm2 = accumarray(faces(:), repmat(triangle_cm2 / 3, 3, 1), [n_vertices, 1]);
if ~all(isfinite(vertex_cm2)) || sum(vertex_cm2) <= 0
    error('DSTCedNet:DegenerateMesh', 'Cortex must have positive finite surface area.');
end
if isfield(cortex, 'VertConn') && ~isempty(cortex.VertConn)
    adjacency = sparse(cortex.VertConn ~= 0);
    if ~isequal(size(adjacency), [n_vertices, n_vertices])
        error('DSTCedNet:InvalidAdjacency', 'VertConn dimensions must match Vertices.');
    end
    adjacency = spones(adjacency + adjacency');
else
    edge_i = [faces(:, 1); faces(:, 2); faces(:, 3)];
    edge_j = [faces(:, 2); faces(:, 3); faces(:, 1)];
    adjacency = sparse([edge_i; edge_j], [edge_j; edge_i], 1, n_vertices, n_vertices);
    adjacency = spones(adjacency);
end
adjacency = adjacency - spdiags(diag(adjacency), 0, n_vertices, n_vertices);

prior_rng = rng;
restore_rng = onCleanup(@() rng(prior_rng)); %#ok<NASGU>
rng(double(seed), 'twister');
W = false(n_vertices, n_sources, n_samples);
centers = zeros(n_samples, n_sources);
achieved_cm2 = zeros(n_samples, n_sources);
requested_cm2 = zeros(n_samples, n_sources);
max_attempts = 1000;

for sample = 1:n_samples
    occupied = false(n_vertices, 1);
    for component = 1:n_sources
        if isempty(condition_areas_cm2)
            target = min_area_cm2 + (max_area_cm2 - min_area_cm2) * rand;
        else
            target = condition_areas_cm2(component);
        end
        if target > sum(vertex_cm2)
            error('DSTCedNet:AreaTooLarge', 'Requested %.6g cm2 exceeds mesh area %.6g cm2.', target, sum(vertex_cm2));
        end
        found = false;
        for attempt = 1:max_attempts
            center = randi(n_vertices);
            [patch, patch_area] = grow_patch(center, adjacency, vertex_cm2, target);
            if patch_area + eps(max(1, target)) < target
                continue; % Seed belongs to an insufficient connected component.
            end
            if ~allow_overlap && any(patch & occupied)
                continue;
            end
            found = true;
            break;
        end
        if ~found
            error('DSTCedNet:SamplingFailed', ['Could not draw source %d for sample %d after %d attempts. ', ...
                'Reduce area/source count or allow overlaps.'], component, sample, max_attempts);
        end
        W(:, component, sample) = patch;
        centers(sample, component) = center;
        achieved_cm2(sample, component) = patch_area;
        requested_cm2(sample, component) = target;
        occupied = occupied | patch;
    end
end
end

function [patch, area_cm2] = grow_patch(center, adjacency, vertex_cm2, target)
% Breadth-first traversal grows one connected neighborhood. Random neighbor
% order avoids a persistent preference for low-index boundary vertices.
n_vertices = size(adjacency, 1);
queue = zeros(n_vertices, 1);
queued = false(n_vertices, 1);
patch = false(n_vertices, 1);
queue(1) = center;
queued(center) = true;
head = 1;
tail = 1;
area_cm2 = 0;
while head <= tail
    vertex = queue(head);
    head = head + 1;
    patch(vertex) = true;
    area_cm2 = area_cm2 + vertex_cm2(vertex);
    if area_cm2 >= target && area_cm2 > 0
        return;
    end
    neighbors = find(adjacency(vertex, :));
    neighbors = neighbors(~queued(neighbors));
    if numel(neighbors) > 1
        neighbors = neighbors(randperm(numel(neighbors)));
    end
    for neighbor = neighbors
        tail = tail + 1;
        queue(tail) = neighbor;
        queued(neighbor) = true;
    end
end
end
