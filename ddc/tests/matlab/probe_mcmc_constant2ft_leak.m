% Minimal semantic probe of the unmodified DDC_MCMC branch-3 assignments.
% These statements are copied in source order without adding rejection
% rollback. They isolate value-copy semantics from stochastic numerics.
matlab_test_dir = fileparts(mfilename('fullpath'));

Constant2ft = [0 0];
Constant2 = Constant2ft;       % outer-iteration source assignment

% First branch-3 attempt: snapshot starting state, then perturb. The produced
% conformation is assumed unchanged, so while isequal(loc,loc2) retries.
Constant2ft = Constant2;
first_snapshot = Constant2ft;
Constant2(1) = Constant2(1) + .1;
after_first_perturbation = Constant2;

% Second branch-3 attempt: source snapshots the intermediate proposal before
% applying its next perturbation.
Constant2ft = Constant2;
second_snapshot = Constant2ft;
Constant2(2) = Constant2(2) - .1;
final_proposal = Constant2;

% The later Metropolis condition rejects. In unmodified DDC_MCMC there is no
% assignment restoring Constant2ft in the rejection path.
proposal_accepted = false;
if proposal_accepted
    Constant2ft = Constant2;
end
constant2ft_after_rejection = Constant2ft;

% Next outer iteration begins from the persistent leaked snapshot.
Constant2_next_iteration = Constant2ft;

save(fullfile(matlab_test_dir, 'mcmc_constant2ft_leak_probe.mat'), ...
    'first_snapshot', 'after_first_perturbation', 'second_snapshot', ...
    'final_proposal', 'proposal_accepted', ...
    'constant2ft_after_rejection', 'Constant2_next_iteration');
