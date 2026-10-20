% Generate a deterministic oracle for Determine_Blinking_Distribution5.
% This calls the unmodified repository function and saves all six outputs.

script_path = mfilename('fullpath');
matlab_test_dir = fileparts(script_path);
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

index = (0:11)';
LocalizationsFinal = {[index * 2, mod(index, 3)]};
% Use a supported double frame fixture for the successful numerical oracle.
% uint16 behavior is probed separately by probe_uint16_frame_pdist.m.
Frame_Information = {double(mod(index, 6) + 1)'};
Pre_A = 2;
Resolution_input = 1;

[bins_expected, blink_expected, total_no_blink_expected, ...
    resolution_expected, x_overall_expected, m_mat_expected] = ...
    Determine_Blinking_Distribution5( ...
        LocalizationsFinal, Frame_Information, Pre_A, Resolution_input);

save(fullfile(matlab_test_dir, 'distribution_reference.mat'), ...
    'LocalizationsFinal', 'Frame_Information', 'Pre_A', 'Resolution_input', ...
    'bins_expected', 'blink_expected', 'total_no_blink_expected', ...
    'resolution_expected', 'x_overall_expected', 'm_mat_expected', '-v7');
