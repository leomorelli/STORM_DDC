% Minimal runtime oracle for DDC-SOURCE-BUG-002.

script_path = mfilename('fullpath');
matlab_test_dir = fileparts(script_path);

frame = uint16([1 3 7]);
X = [[1:length(frame)]' * 0, frame(:)];
X_class = class(X);
pdist_succeeded = true;
pdist_output = [];
pdist_error_identifier = '';
pdist_error_message = '';

try
    pdist_output = pdist(X);
catch error
    pdist_succeeded = false;
    pdist_error_identifier = error.identifier;
    pdist_error_message = error.message;
end

save(fullfile(matlab_test_dir, 'uint16_frame_pdist_probe.mat'), ...
    'frame', 'X', 'X_class', 'pdist_succeeded', 'pdist_output', ...
    'pdist_error_identifier', 'pdist_error_message', '-v7');

disp(['class(X): ', X_class]);
disp(['pdist succeeded: ', mat2str(pdist_succeeded)]);
if pdist_succeeded
    disp(pdist_output);
else
    disp(pdist_error_identifier);
    disp(pdist_error_message);
end
