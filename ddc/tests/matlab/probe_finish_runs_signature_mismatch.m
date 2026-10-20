% Oracle for DDC-SOURCE-BUG-007. Run from the repository root.
root = pwd;
ddc_path = fullfile(root, 'Main_DDC_Folder', 'Scripts_Functions');
cluster_finish = fullfile(root, 'Main_DDC_Folder', ...
    'Scripts_To_Use_on_Cluster', 'Finish_Runs.m');
local_finish = fullfile(ddc_path, 'Finish_Runs.m');
addpath(ddc_path);

resolved_ddc_mcmc = which('DDC_MCMC');
ddc_mcmc_nargout = nargout('DDC_MCMC');
assert(ddc_mcmc_nargout == 12);
assert(isfile(local_finish));
assert(isfile(cluster_finish));

% MATLAB assigns the first N declared outputs, in declaration order, when a
% caller requests fewer outputs. Omitting the new leading Traj2 therefore
% shifts every stale Finish_Runs assignment by one.
[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11] = twelve_outputs();
assigned_first_eleven = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11];
assert(isequal(assigned_first_eleven, 1:11));
finish_runs_requested_outputs = 11;
finish_runs_omits_traj2 = true;
prob_dists2_actual_output_name = 'Orderf';
constant2ft_captured = false;
post_return_failure_pending = true;

save(fullfile(fileparts(mfilename('fullpath')), ...
    'finish_runs_signature_reference.mat'), ...
    'resolved_ddc_mcmc', 'ddc_mcmc_nargout', 'local_finish', ...
    'cluster_finish', 'assigned_first_eleven', ...
    'finish_runs_requested_outputs', 'finish_runs_omits_traj2', ...
    'prob_dists2_actual_output_name', 'constant2ft_captured', ...
    'post_return_failure_pending', '-v7');

function [a,b,c,d,e,f,g,h,i,j,k,l] = twelve_outputs()
a=1; b=2; c=3; d=4; e=5; f=6;
g=7; h=8; i=9; j=10; k=11; l=12;
end
