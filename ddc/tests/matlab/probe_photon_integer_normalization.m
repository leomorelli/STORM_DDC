% Probe the exact active expression weights/sum(weights), without casting.
% Run from this directory. The small non-v7.3 MAT is intentionally excluded
% from review archives and is consumed only as an optional Python oracle.

types = {'double','single','uint8','uint8','uint8','uint16','int8','int16', ...
         'double','single','uint8','uint16','int8','int16'};
values = {[1 3],[1 3],[1 1],[1 1 1],[1 3],[1 3],[1 3],[1 3], ...
          [0 0],[0 0],[0 0],[0 0],[-1 1],[-1 1]};

records = cell(numel(types),1);
for k = 1:numel(types)
    weights = cast(values{k}, types{k});
    record.original = weights;
    record.weights_class = class(weights);
    record.total = sum(weights);
    record.total_class = class(record.total);
    try
        record.normalized = weights/sum(weights);
        record.normalized_class = class(record.normalized);
        record.succeeded = true;
        record.error_identifier = '';
        record.error_message = '';
    catch ME
        record.normalized = [];
        record.normalized_class = '';
        record.succeeded = false;
        record.error_identifier = ME.identifier;
        record.error_message = ME.message;
    end
    records{k} = record;
    clear record
end

% Also record the next source operation with normal double coordinates.
coordinates = [0.25; 2.75];
integer_weights = uint8([1; 3]);
integer_weights = integer_weights/sum(integer_weights);
coordinate_product_succeeded = true;
coordinate_product = [];
coordinate_product_class = '';
coordinate_product_error_identifier = '';
coordinate_product_error_message = '';
try
    coordinate_product = coordinates .* integer_weights;
    coordinate_product_class = class(coordinate_product);
catch ME
    coordinate_product_succeeded = false;
    coordinate_product_error_identifier = ME.identifier;
    coordinate_product_error_message = ME.message;
end

% Floating mixed-class source-operation probes. No operand is cast before .*
floating_coordinates = {double([1;2]), single([1;2]), ...
                        single([1;2]), double([1;2])};
floating_weights = {single([0.25;0.75]), double([0.25;0.75]), ...
                    single([0.25;0.75]), double([0.25;0.75])};
floating_products = cell(4,1);
for k = 1:4
    probe.coordinate_class = class(floating_coordinates{k});
    probe.weight_class = class(floating_weights{k});
    probe.product_succeeded = true;
    probe.product = [];
    probe.product_class = '';
    probe.product_sum = [];
    probe.product_sum_class = '';
    probe.error_identifier = '';
    probe.error_message = '';
    try
        probe.product = floating_coordinates{k} .* floating_weights{k};
        probe.product_class = class(probe.product);
        probe.product_sum = sum(probe.product);
        probe.product_sum_class = class(probe.product_sum);
    catch ME
        probe.product_succeeded = false;
        probe.error_identifier = ME.identifier;
        probe.error_message = ME.message;
    end
    floating_products{k} = probe;
    clear probe
end

% The scalar-double exception for integer arithmetic is distinct from the
% retained nonscalar double-vector/uint8 failure above.
scalar_double = double(300);
scalar_uint8 = uint8(1);
scalar_integer_product_succeeded = true;
scalar_integer_product = [];
scalar_integer_product_class = '';
scalar_integer_product_error_identifier = '';
scalar_integer_product_error_message = '';
try
    scalar_integer_product = scalar_double .* scalar_uint8;
    scalar_integer_product_class = class(scalar_integer_product);
catch ME
    scalar_integer_product_succeeded = false;
    scalar_integer_product_error_identifier = ME.identifier;
    scalar_integer_product_error_message = ME.message;
end

save('photon_integer_normalization_reference.mat', 'records', ...
     'coordinate_product_succeeded', 'coordinate_product', ...
     'coordinate_product_class', 'coordinate_product_error_identifier', ...
     'coordinate_product_error_message', 'floating_products', ...
     'scalar_integer_product_succeeded', 'scalar_integer_product', ...
     'scalar_integer_product_class', ...
     'scalar_integer_product_error_identifier', ...
     'scalar_integer_product_error_message', '-v7');
