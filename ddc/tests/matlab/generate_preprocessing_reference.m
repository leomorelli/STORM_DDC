function generate_preprocessing_reference
% Developer-only numerical oracle. Python tools do not require MATLAB.
% Evaluate the computation sections of the original scripts without GUI I/O.
folder = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(folder));
source = fullfile(root, 'DDC-main', 'Main_DDC_Folder', 'DDC_Tools');
rng(16);
centers = rand(45, 2) * 4000;
loc = repelem(centers, 4, 1) + randn(180, 2) * 45;
frame = repelem((0:44)' * 20, 4) + repmat((0:3)', 45, 1);
LocalizationsFinal = {loc, loc * 1.1};
Frame_Information = {frame, frame};
TrueLocalizations = {};

NFTP = 81; Gap = 10;
text = fileread(fullfile(source, 'Determine_N.m'));
first = strfind(text, 'Cum_Sum_Store=[];');
last = strfind(text, '%% Plotting Z');
evalc(text(first(1):last(1)-1));
cdf_expected = Cum_Sum_Store; z_expected = Z;

Res = 80; N_f = 10; image_frac = 1;
text = fileread(fullfile(source, 'Determine_Res.m'));
first = strfind(text, 'D_maxi=0;');
last = strfind(text, '%% Ploting TPDD');
evalc(text(first(1):last(1)-1));
res_expected = Res_Store;

text = fileread(fullfile(source, 'New_Determine_Res.m'));
first = strfind(text, 'Pre_A=N_f;');
evalc(text(first(1):end));
resolution_expected = Resolution; blink_expected = D_Counts3;
save(fullfile(folder, 'preprocessing_reference.mat'), 'LocalizationsFinal', ...
    'Frame_Information', 'cdf_expected', 'z_expected', 'res_expected', ...
    'resolution_expected', 'blink_expected', '-v7');
fprintf('Saved preprocessing_reference.mat (resolution %g nm)\n', Resolution);
generate_split(source, folder, LocalizationsFinal, Frame_Information);
end

function generate_split(source, folder, LocalizationsFinal, Frame_Information)
% Replay the same uniform random values in Python to test the partition search,
% quantiles, buffer adjustment, ordering, and preserved frame/photon alignment.
rng(7); random_values = rand(1, 20000); rng(7);
TrueLocalizations = {LocalizationsFinal{1}(1:4:end,:), LocalizationsFinal{2}(1:4:end,:)};
Photons = {(1:180)', (1:180)'};
min_loc = 60;
text = fileread(fullfile(source, 'Split_UP_Image_Multiple_Images.m'));
first = strfind(text, '%If photons cell does not exist');
last = strfind(text, "save(['Split_',Condition]");
text = text(first(1):last(1)-1);
text = strrep(text, 'while onwers<300', 'while onwers<15');
text = regexprep(text, '(?m)^\s*(figure\(1\)|scatter3\([^\r\n]*|axis equal|drawnow|colormap jet|pause\(\.5\))\s*$', '');
evalc(text);
save(fullfile(folder, 'split_preprocessing_reference.mat'), 'LocalizationsFinal', ...
    'Frame_Information', 'Photons', 'TrueLocalizations', 'Came_from_image', ...
    'addonarray', 'Parameters_to_split', 'cut1array', 'cut2array', 'cut3array', ...
    'random_values', '-v7');
fprintf('Saved split_preprocessing_reference.mat (%d regions)\n', length(LocalizationsFinal));
end
