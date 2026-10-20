% Probe DDC-SOURCE-BUG-004 using the unmodified Density_Calc script.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
scripts_dir = fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions');

LocalizationsFinal = [0 0; 0 0];
Frame_Information = uint16([1 3]);
Resolution = 1;
A = 1;
forward_difference = abs(Frame_Information(1)-Frame_Information(2));
reverse_difference = abs(Frame_Information(2)-Frame_Information(1));
run(fullfile(scripts_dir, 'Density_Calc.m'));
density_expected = density;
density_class = class(density);
frame_class = class(Frame_Information);

save(fullfile(matlab_test_dir, 'uint16_density_probe.mat'), ...
    'LocalizationsFinal', 'Frame_Information', 'Resolution', 'A', ...
    'forward_difference', 'reverse_difference', 'density_expected', ...
    'density_class', 'frame_class');
