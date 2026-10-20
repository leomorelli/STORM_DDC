% Execute the unmodified Calculate_Score_8 script in a minimal workspace.
matlab_test_dir = fileparts(mfilename('fullpath'));
repository_root = fileparts(fileparts(matlab_test_dir));
scripts_dir = fullfile(repository_root, 'Main_DDC_Folder', 'Scripts_Functions');

loc = [0 0; 1 0];
fram = double([1 2]);
A = 4;
D_Overall = [1 3 2];
Z2 = [1 3 2];
Resolution = 1;
Prob_Distributions = [10 11 12 13; 20 21 22 23; ...
    30 31 32 33; 40 41 42 43];

loc_input = loc;
fram_input = fram;
A_input = A;
D_Overall_input = D_Overall;
Z2_input = Z2;
Resolution_input = Resolution;
Prob_Distributions_input = Prob_Distributions;
run(fullfile(scripts_dir, 'Calculate_Score_8.m'));

D_expected = D;
Z22_expected = Z22;
C_expected = C;
ia_expected = ia;
ib_expected = ib;
D_temp_expected = D_temp;
Z2_temp_expected = Z2_temp;
lik_expected = lik;
lik2_expected = lik2;
D_class = class(D);
Z22_class = class(Z22);
C_class = class(C);
ia_class = class(ia);
ib_class = class(ib);
D_temp_class = class(D_temp);
Z2_temp_class = class(Z2_temp);
lik_class = class(lik);
lik2_class = class(lik2);
D_size = size(D);
Z22_size = size(Z22);
C_size = size(C);
ia_size = size(ia);
ib_size = size(ib);
D_temp_size = size(D_temp);
Z2_temp_size = size(Z2_temp);
lik_size = size(lik);
lik2_size = size(lik2);

save(fullfile(matlab_test_dir, 'likelihood_reference.mat'), ...
    'loc_input', 'fram_input', 'A_input', 'D_Overall_input', 'Z2_input', ...
    'Resolution_input', 'Prob_Distributions_input', 'D_expected', ...
    'Z22_expected', 'C_expected', 'ia_expected', 'ib_expected', ...
    'D_temp_expected', 'Z2_temp_expected', 'lik_expected', 'lik2_expected', ...
    'D_class', 'Z22_class', 'C_class', 'ia_class', 'ib_class', ...
    'D_temp_class', 'Z2_temp_class', 'lik_class', 'lik2_class', ...
    'D_size', 'Z22_size', 'C_size', 'ia_size', 'ib_size', ...
    'D_temp_size', 'Z2_temp_size', 'lik_size', 'lik2_size');
