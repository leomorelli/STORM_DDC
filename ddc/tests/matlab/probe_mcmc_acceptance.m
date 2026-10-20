% Deterministic oracle for the exact DDC_MCMC Metropolis expression.
new_lik = [1, 0, -1, -1];
current_lik = [0, 0, 0, 0];
controlled_rand = [0.9, 0.9, 0.2, 0.8];
accepted = false(size(new_lik));

for index = 1:numel(new_lik)
    accepted(index) = new_lik(index) >= current_lik(index) || ...
        log(controlled_rand(index)) <= new_lik(index)-current_lik(index);
end

assert(isequal(accepted, logical([1, 1, 1, 0])));

steps = 42;
max_lik = 0;
strict_new_maximum = 1;
if strict_new_maximum > max_lik
    max_lik = strict_new_maximum;
    steps = 10;
end
steps_after_maximum = steps;

save(fullfile(fileparts(mfilename('fullpath')), ...
    'mcmc_acceptance_probe.mat'), 'new_lik', 'current_lik', ...
    'controlled_rand', 'accepted', 'steps_after_maximum', '-v7');
