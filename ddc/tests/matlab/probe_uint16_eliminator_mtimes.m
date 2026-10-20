% Probe DDC-SOURCE-BUG-003 without casting repository-typed frames.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

locations = [0 0; 1 0];
frame = uint16([1 2]);
resolution = 1;
A = 2;
deviation = ones(2, 2);
traj_num = [1 2];
AAAA = frame(1);
AAAA = [AAAA(:), AAAA(:)*0];
BBBB = frame(2);
BBBB = [BBBB(:), BBBB(:)*0];
AAAA_class = class(AAAA);
BBBB_class = class(BBBB);

mtimes_succeeded = true;
mtimes_error = '';
try
    product = AAAA*BBBB'; %#ok<NASGU>
catch ME
    mtimes_succeeded = false;
    mtimes_error = ME.message;
end

function_succeeded = true;
function_error = '';
try
    Eliminate_Blinking_De_Loc15_MCMC( ...
        locations, frame, resolution, A, deviation, traj_num);
catch ME
    function_succeeded = false;
    function_error = ME.message;
end

save(fullfile(matlab_test_dir, 'uint16_eliminator_mtimes_probe.mat'), ...
    'frame', 'AAAA_class', 'BBBB_class', 'mtimes_succeeded', ...
    'mtimes_error', 'function_succeeded', 'function_error');
