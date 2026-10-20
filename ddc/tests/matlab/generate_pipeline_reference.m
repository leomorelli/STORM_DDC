% Deterministic Stage A.8 wrapper around active run_script_DDC statements.
% GUI, Run_cluster path selection, parfor, MCMC trajectories and save calls
% are intentionally excluded from this in-memory oracle.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

LocalizationsFinal = {[0 0; 10 20; 30 40; 50 60]};
Frame_Information = {double([1; 2; 3; 4])};
TrueLocalizations = [];
Photons = {[]};

if isempty(Photons{1})
    for i=1:length(Frame_Information)
        vv1=ones(1,length(Frame_Information{i}));
        Photons{i}=vv1(:);
    end
end
default_photons_expected = Photons{1};

if isempty(TrueLocalizations)
    TrueLocalizations={};
    for ijk=1:length(Frame_Information)
        TrueLocalizations{ijk}=[];
    end
end
empty_truth_size_expected = size(TrueLocalizations{1});

for sdfv=1:length(LocalizationsFinal)
    if min(size(LocalizationsFinal{sdfv}))<3
        LocalizationsFinal{sdfv}(:,3)=LocalizationsFinal{sdfv}(:,2)*0;
    end
end
padded_2d_expected = LocalizationsFinal{1};

Orderf={[]}; Lik={[]}; RelScore={[]}; Numb_of_Loc={[]};
Constantf={[]}; Constantf2={[]}; Step=0;
Prob_dists{1}.Deviation_in_Probability=[];
Prob_dists{1}.Prob_Distributions=[];
Prob_dists{1}.Dscale_store=[];
initialized_cell_sizes = [length(Orderf), length(Lik), length(RelScore), ...
    length(Numb_of_Loc), length(Constantf), length(Constantf2), ...
    length(Prob_dists)];

eligibility_length_10 = length(zeros(10,3))>10;
eligibility_length_11 = length(zeros(11,3))>10;

% Deterministic photon-weighted source loop with supplied trajectory labels.
weighted_input = [0 0 1; 10 20 3];
weighted_frames = [1; 5];
weighted_photons = [1; 3];
Traj2 = [2 2];
New_final_loc=[]; New_final_frame=[];
for isev=unique(Traj2)
    weights=weighted_photons(Traj2==isev);
    weights=weights/sum(weights);
    New_final_loc=[New_final_loc; [ ...
        sum(weighted_input(Traj2==isev,1).*weights(:)), ...
        sum(weighted_input(Traj2==isev,2).*weights(:)), ...
        sum(weighted_input(Traj2==isev,3).*weights(:))]];
    New_final_frame(end+1)=mean(weighted_frames(Traj2==isev));
end
weighted_locations_expected = New_final_loc;
weighted_frames_expected = New_final_frame;

old_history=[]; new_history=[1 .5];
maxtemp=max(old_history);
if isempty(maxtemp); maxtemp=-Inf; end
first_replacement_expected = maxtemp<max(new_history);
old_history=[1 .5]; new_history=[1 .5 .25];
no_improvement_expected = max(old_history)<max(new_history);

% Supported double-frame shared distribution fixture. It is evaluated once,
% before any per-image MCMC call, as in run_script_DDC.
frames_distribution = [ones(1,11), 2, ones(1,11)*4];
x_distribution = [(0:10)*10, .2, (0:10)*10+.5];
distribution_locations = {[x_distribution(:), zeros(23,1)]};
distribution_frames = {double(frames_distribution)};
[bins_expected, Distribution_for_Blink_expected, ~, ...
    Resolution_expected, X_overall_expected, M_mat_expected] = ...
    Determine_Blinking_Distribution5(distribution_locations, ...
    distribution_frames, 2, 10);

save(fullfile(matlab_test_dir, 'pipeline_reference.mat'), ...
    'default_photons_expected', 'empty_truth_size_expected', ...
    'padded_2d_expected', 'initialized_cell_sizes', ...
    'eligibility_length_10', 'eligibility_length_11', ...
    'weighted_locations_expected', 'weighted_frames_expected', ...
    'first_replacement_expected', 'no_improvement_expected', ...
    'bins_expected', 'Distribution_for_Blink_expected', ...
    'Resolution_expected', 'X_overall_expected', 'M_mat_expected');
