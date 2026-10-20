% Run the unmodified Stage A.5 scripts in minimal faithful workspaces.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
scripts_dir = fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions');

LocalizationsFinal = [0 0 0; 10 0 0; 0 0 1; 10 0 1];
Frame_Information = double([5 2 5 2]);
Resolution = 1;
run(fullfile(scripts_dir, 'Determine_Locs_w_Multi2.m'));
multi_locations = LocalizationsFinal;
multi_frames = Frame_Information;
multi_resolution = Resolution;
frames_w_multi_expected = Frames_w_Multi;
frames_w_multi_class = class(Frames_w_Multi);
frames_w_multi_size = size(Frames_w_Multi);

clear LocalizationsFinal Frame_Information Resolution Frames_w_Multi
LocalizationsFinal = [0 0; 2 0; 10 0];
Frame_Information = double([1 3 8]);
Resolution = 1;
A = 1;
run(fullfile(scripts_dir, 'Density_Calc.m'));
density_locations = LocalizationsFinal;
density_frames = Frame_Information;
density_resolution = Resolution;
density_a = A;
distance_dens_expected = DistanceDens;
density_expected = density;
density_class = class(density);
density_size = size(density);

clear LocalizationsFinal Frame_Information Resolution A density DistanceDens
LocalizationsFinal = [0 0; 10 0];
Frame_Information = double([1 1]);
Resolution = 1;
A = 1;
run(fullfile(scripts_dir, 'Density_Calc.m'));
constant_locations = LocalizationsFinal;
constant_frames = Frame_Information;
constant_resolution = Resolution;
constant_a = A;
constant_density_expected = density;
constant_density_class = class(density);
constant_density_size = size(density);

clear LocalizationsFinal Frame_Information Resolution A density DistanceDens
LocalizationsFinal = [0 0 0; 1 0 0];
Frame_Information = double([1 3]);
Resolution = 1;
A = 1;
n_less_d_succeeded = true;
n_less_d_error = '';
try
    run(fullfile(scripts_dir, 'Density_Calc.m'));
catch ME
    n_less_d_succeeded = false;
    n_less_d_error = ME.message;
end

save(fullfile(matlab_test_dir, 'trajectory_reference.mat'), ...
    'multi_locations', 'multi_frames', 'multi_resolution', ...
    'frames_w_multi_expected', 'frames_w_multi_class', ...
    'frames_w_multi_size', 'density_locations', 'density_frames', ...
    'density_resolution', 'density_a', 'distance_dens_expected', ...
    'density_expected', 'density_class', 'density_size', ...
    'constant_locations', 'constant_frames', 'constant_resolution', ...
    'constant_a', 'constant_density_expected', 'constant_density_class', ...
    'constant_density_size', 'n_less_d_succeeded', 'n_less_d_error');
