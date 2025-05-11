
function saveCellArray3D(cellArray, filename)
% saveCellArray3D Save a 1xN cell array of Nx3 matrices as a tab-delimited text file.
% Each cell's X,Y,Z columns are stacked side-by-side.
%
% Usage:
%   saveCellArray3D(cellArray, 'output.txt')
%
% Inputs:
%   - cellArray: 1xN cell array, each cell containing an Mx3 numeric matrix
%   - filename:  Name of the output text file (e.g., 'data.txt')

    if ~iscell(cellArray)
        error('Input must be a cell array.');
    end

    numCells = length(cellArray);
    maxRows = max(cellfun(@(x) size(x,1), cellArray));

    % Preallocate with NaNs to handle variable row sizes
    combined = NaN(maxRows, numCells * 3);

    for i = 1:numCells
        data = cellArray{i};
        if size(data,2) ~= 3
            error('Each cell must contain a Nx3 matrix.');
        end
        [rows, ~] = size(data);
        colStart = (i - 1)*3 + 1;
        combined(1:rows, colStart:colStart+2) = data;
    end

    % Write to file
    writematrix(combined, filename, 'Delimiter', 'tab');
    fprintf('Saved successfully to: %s\n', filename);
end


[filename, pathname] = uigetfile({'*.mat'}, 'Select HMM .mat file');
if ( filename == 0 )
    disp('Error! No (or wrong) file selected!')
    return
end

%This will load in all of the localizations from your structure file
full_filename = [ pathname, filename ];
load(full_filename);

Condition=filename;



dlmwrite('cut1array_3d.txt', cut1array, 'delimiter', '\t')
dlmwrite('cut2array_3d.txt', cut2array, 'delimiter', '\t')
dlmwrite('cut3array_3d.txt', cut3array, 'delimiter', '\t')
dlmwrite('camefromimage_3d.txt', Came_from_image, 'delimiter', '\t')
dlmwrite('Frame_Information_3d.txt', Frame_Information, 'delimiter', '\t')
saveCellArray3D(TrueLocalizations, 'Truelocalizations_3d.txt');
saveCellArray3D(LocalizationsFinal, 'LocalizationsFinal_3d.txt');
dlmwrite('addonarray_3d.txt', addonarray, 'delimiter', '\t')
dlmwrite('parameters_to_split_3d.txt', Parameters_to_split, 'delimiter', '\t')
