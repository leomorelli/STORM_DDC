function generate_combine_reference
% Developer-only oracle for the original MATLAB boundary behavior.
folder = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(folder));
source = fullfile(root, 'DDC-main', 'Main_DDC_Folder', 'DDC_Tools', 'Combine_Images.m');
points = [-1 0 0; 0 0 0; 0.5 0.5 0; 1 0 0; 1 0.5 0; 1 1 0; ...
          1.5 0.5 0; 2 1 0; 3 0 0; 1 1 -1; 1 1 1; 0 0 0];
Final_Loc_Blinking_Corrected = repmat({points}, 1, 6);
Final_Frame_Blinking_Corrected = repmat({(1:size(points,1))'}, 1, 6);
Came_from_image = [1 1 2 2 3 3];
cut1array = {[0 1], [1 2], [0 1], [1 2], [0 2], [0 2]};
cut2array = {[0 1], [0 1], [0 1], [0 1], [0 2], [0 2]};
cut3array = {[0 0], [0 0], [0 0], [0 0], [-1 0], [0 1]};
addonarray = zeros(1, 6);
save(fullfile(folder, 'combine_reference_input.mat'), ...
     'Final_Loc_Blinking_Corrected', 'Final_Frame_Blinking_Corrected', ...
     'Came_from_image', 'cut1array', 'cut2array', 'cut3array', 'addonarray', '-v7');
text = fileread(source);
first = strfind(text, 'LocalizationsFinal={};');
last = strfind(text, "save(['Combined_Final_',Condition]");
text = text(first(1):last(1)-1);
text = regexprep(text, '(?m)^\s*(figure\(1\)|scatter3\([^\r\n]*|axis equal|drawnow|colormap jet|pause\(\.5\))\s*$', '');
Cutsstore = [];
evalc(text);
save(fullfile(folder, 'combine_reference_expected.mat'), ...
     'Final_Loc_Blinking_Corrected', 'Final_Frame_Blinking_Corrected', '-v7');
fprintf('Saved combiner reference for %d original images\n', length(Final_Loc_Blinking_Corrected));
end
