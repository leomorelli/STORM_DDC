% Execute unmodified PALMplot numerical code. Active source creates no figure.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
addpath(fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions'));

inferred_coords = [0 0; 50 0];
true_coords = [0 0; 100 0];
inferred_image_expected = PALMplot(inferred_coords, true_coords);
true_image_expected = PALMplot(true_coords, true_coords);
score_expected = sum(sum(abs(mat2gray(inferred_image_expected) - ...
    mat2gray(true_image_expected)).^2));
inferred_image_class = class(inferred_image_expected);
true_image_class = class(true_image_expected);
inferred_image_size = size(inferred_image_expected);
true_image_size = size(true_image_expected);

save(fullfile(matlab_test_dir, 'scoring_reference.mat'), ...
    'inferred_coords', 'true_coords', 'inferred_image_expected', ...
    'true_image_expected', 'score_expected', 'inferred_image_class', ...
    'true_image_class', 'inferred_image_size', 'true_image_size');
