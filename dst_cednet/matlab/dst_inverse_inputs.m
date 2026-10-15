function [X, L] = dst_inverse_inputs(sensors, gain)
%DST_INVERSE_INPUTS Validate one trial for the maintained inverse algorithms.
% X [sensors,time], L [sensors,vertices]; all returned values are double.
% Copyright (c) 2026 DST-CedNet contributors. MIT License.
validateattributes(sensors, {'numeric'}, {'2d','real','finite','nonempty'});
validateattributes(gain, {'numeric'}, {'2d','real','finite','nonempty'});
X = double(sensors);
L = double(gain);
if size(X, 1) ~= size(L, 1)
    error('DSTCedNet:SensorShape', 'Sensor rows must match lead-field rows.');
end
if ~any(L(:))
    error('DSTCedNet:ZeroLeadField', 'The lead field must contain a nonzero column.');
end
end
