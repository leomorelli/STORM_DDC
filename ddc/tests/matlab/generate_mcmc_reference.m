% Deterministic/source-contract oracle for DDC_MCMC initialization.
% Run from the repository root. This does not claim MATLAB/NumPy RNG identity.
rng(1, 'twister');

input_frames = [ones(1,11), 2, 4*ones(1,11)];
x = [(0:10)*10, 0.2, (0:10)*10 + 0.5];
input_localizations = [x(:), zeros(numel(x),1)];
A = 3;

[~, Inds] = sort(input_frames);
sorted_frames = input_frames(Inds);
sorted_localizations = input_localizations(Inds,:);
frame_is_sorted = issorted(sorted_frames);

Z2 = pdist([[1:length(sorted_frames)]'*0, sorted_frames(:)]);
good = false;
while ~good
    jittered_localizations = sorted_localizations + ...
        rand(size(sorted_localizations))*.01;
    D_Overall = pdist(jittered_localizations);
    selected_z2 = Z2(Z2 < A);
    D_Overall = D_Overall(Z2 < A);
    D_Overall = D_Overall(selected_z2 > 0);
    d_overall_is_unique = length(unique(D_Overall)) == length(D_Overall);
    good = d_overall_is_unique;
end

% These are the deterministic shapes/classes established before proposals.
Final_Localizations_Blinking_Corrected = jittered_localizations;
Final_Frame_Blinking_Corrected = sorted_frames;
input_localizations_class = class(input_localizations);
input_frames_class = class(input_frames);
output_localizations_class = class(Final_Localizations_Blinking_Corrected);
output_frames_class = class(Final_Frame_Blinking_Corrected);
initial_order = 1:length(sorted_frames);
initial_constant = zeros(1,length(sorted_frames));
localization_shape = size(Final_Localizations_Blinking_Corrected);
frame_shape = size(Final_Frame_Blinking_Corrected);

save(fullfile(fileparts(mfilename('fullpath')), ...
    'mcmc_initialization_reference.mat'), ...
    'input_localizations', 'input_frames', ...
    'Final_Localizations_Blinking_Corrected', ...
    'Final_Frame_Blinking_Corrected', 'frame_is_sorted', ...
    'd_overall_is_unique', 'input_localizations_class', ...
    'input_frames_class', 'output_localizations_class', ...
    'output_frames_class', 'initial_order', 'initial_constant', ...
    'localization_shape', 'frame_shape', '-v7');
