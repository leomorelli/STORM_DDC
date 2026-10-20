% Generate deterministic Stage A.4 oracle values with supported double frames.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

locations = [0 0; 1 0; 3 0];
frames = double([1 2 3]);
resolution = 1;
A = 2;
deviation = [.8 .8; 0 0];
traj_num = [3 1 2];

[loc_expected, probs_expected] = Eliminate_Blinking_De_Loc15_MCMC( ...
    locations, frames, resolution, A, deviation, traj_num);

true_distribution = [0 1];
blink_distribution = [1 0];
bins = [0 1 Inf];
true_distribution2 = [0 1];
blink_distribution2 = [1 0];
bins2 = [0 1.5 Inf];
x_overall = [0 1];
m_mat = deviation;

rng(1);
[deviation_expected, probability_expected, dscale_expected] = ...
    Determine_Deviation_in_Probability8( ...
    true_distribution, blink_distribution, bins, A, locations, frames, ...
    [], [], resolution, true_distribution2, blink_distribution2, bins2, ...
    x_overall, m_mat);

save(fullfile(matlab_test_dir, 'probability_reference.mat'), ...
    'locations', 'frames', 'resolution', 'A', 'deviation', 'traj_num', ...
    'loc_expected', 'probs_expected', 'true_distribution', ...
    'blink_distribution', 'bins', 'true_distribution2', ...
    'blink_distribution2', 'bins2', 'x_overall', 'm_mat', ...
    'deviation_expected', 'probability_expected', 'dscale_expected');
