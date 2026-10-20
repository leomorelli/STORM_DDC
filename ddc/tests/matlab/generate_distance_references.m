% Generate deterministic MATLAB oracle outputs for tests/test_distances.py.
% Run this file with MATLAB from any working directory. It calls the original
% repository function rather than duplicating its algorithm.

script_path = mfilename('fullpath');
matlab_test_dir = fileparts(script_path);
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

two_d_locations = [0 0; 3 4; 0 4; 0 0];
two_d_frames = [1 2 4 4];
two_d_B = 3;
two_d_expected = repmat({[]}, 1, two_d_B + 1);
two_d_expected = Make_Distance_Matrix_Individual( ...
    two_d_locations, two_d_expected, two_d_frames, two_d_B);

three_d_locations = [0 0 12; 0 0 0; 0 4 0];
three_d_frames = [5 1 3];
three_d_B = 4;
three_d_expected = repmat({[]}, 1, three_d_B);
three_d_expected = Make_Distance_Matrix_Individual( ...
    three_d_locations, three_d_expected, three_d_frames, three_d_B);

repeated_locations = [0 0; 3 4; 0 4];
repeated_frames = [2 2 3];
repeated_B = 2;
repeated_expected = repmat({[]}, 1, repeated_B);
repeated_expected = Make_Distance_Matrix_Individual( ...
    repeated_locations, repeated_expected, repeated_frames, repeated_B);

single_locations = [1 2 3];
single_frames = 7;
single_B = 2;
single_expected = repmat({[]}, 1, single_B + 1);
single_expected = Make_Distance_Matrix_Individual( ...
    single_locations, single_expected, single_frames, single_B);

boundary_locations = [0 0; 3 4; 0 12];
boundary_frames = [10 13 14];
boundary_B = 3;
boundary_expected = repmat({[]}, 1, boundary_B + 1);
boundary_expected = Make_Distance_Matrix_Individual( ...
    boundary_locations, boundary_expected, boundary_frames, boundary_B);

unsigned_locations = [0 0; 6 8];
unsigned_frames = uint16([2 4]);
unsigned_B = 2;
unsigned_expected = repmat({[]}, 1, unsigned_B);
unsigned_succeeded = true;
unsigned_error_identifier = '';
try
    unsigned_expected = Make_Distance_Matrix_Individual( ...
        unsigned_locations, unsigned_expected, unsigned_frames, unsigned_B);
catch error
    unsigned_succeeded = false;
    unsigned_error_identifier = error.identifier;
end

save(fullfile(matlab_test_dir, 'distance_reference.mat'), ...
    'two_d_locations', 'two_d_frames', 'two_d_B', 'two_d_expected', ...
    'three_d_locations', 'three_d_frames', 'three_d_B', 'three_d_expected', ...
    'repeated_locations', 'repeated_frames', 'repeated_B', 'repeated_expected', ...
    'single_locations', 'single_frames', 'single_B', 'single_expected', ...
    'boundary_locations', 'boundary_frames', 'boundary_B', 'boundary_expected', ...
    'unsigned_locations', 'unsigned_frames', 'unsigned_B', ...
    'unsigned_expected', 'unsigned_succeeded', 'unsigned_error_identifier', ...
    '-v7');
