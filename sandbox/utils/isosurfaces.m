%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The following code is used to generate isosurfaces for the accuracy values
%   corresponding to same considered parameter epsilon values, for different
%   levels and noise types.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

model = ['resnet'];
noise = 'whole';
levels = [3,5,9,17];
epsValues = [0.03, 0.06, 0.12, 0.24, 0.36]
fileList = {}

for level = levels
    fileList{end+1} = strcat('../',model, '/results/eps_comparison/accuracies_eps_',model,'_',string(level),'_',noise,'_with_comp.csv'); 
end

% noise types are defined in the matlab file: '../data/matlab/4bit.mat' in the str variable
noiseData = load('../data/matlab/4bit.mat');
noiseTypes = noiseData.str
% Add "no Noise" to the noise types at the beginning
noiseTypes = ['no Noise', noiseTypes];

numLevels = length(levels);
numEps = length(epsValues);
numNoiseTypes = length(noiseTypes);
numFiles = length(fileList);
data = zeros(numLevels, numEps, numNoiseTypes);

assert(numLevels == numFiles, 'Number of levels and files do not match');

% Read the data from the files specified in the file list
for fileIdx = 1:length(fileList)
    file = fileList{fileIdx};
    csvData = csvread(file);
    % get rid of first row and first column
    csvData = csvData(2:end,2:end);
    data(fileIdx,:,:) = csvData;
end

% Generate the grid for the data
[x,y] = meshgrid(1:numLevels, 1:numNoiseTypes);
data = permute(data, [3,1,2]);

% For each epsilon value, generate the isosurface
figure;
hold on;

% select a colormap
cmap = spring(numEps);
legendInfo = {};
surfaceHandles = [];

for epsIdx = 1:numEps
    isoValue = epsValues(epsIdx);
    % generate the isosurface
    epsData = data(:,:,epsIdx);
    s = surf(x,y, epsData);
    % set the color of the surface
    set(s, 'FaceColor', cmap(epsIdx,:));
    set(s, 'EdgeColor', 'none');
    % higlight the points on the surface and lines connecting them
    scatter3(x(:), y(:), epsData(:), 'filled', 'MarkerFaceColor', 'black');
    plot3(x', y', epsData', 'Color', 'black')
    plot3(x, y, epsData, 'Color', 'black', 'LineWidth', 1.5)
    % set the transparency of the surface
    alpha(s, 0.7);
    surfaceHandles(end+1) = s;
    legendInfo{end+1} = strcat('Epsilon = ', string(isoValue));
end
hold off;


xlabel('Quantization Levels');
% set ticks only for the specified levels values [3,5,9,17,33]
set(gca, 'XTick', 1:numLevels, 'XTickLabel', levels);
ylabel('Noise Type');
set(gca, 'YTick', 1:numNoiseTypes, 'YTickLabel', noiseTypes);
zlabel('Accuracy');
legend(surfaceHandles,legendInfo);
grid on;
view(3);
camlight;
%lighting gouraud;

% Save the figure
filename = '../',model,'/plots_from_server/', model,'_3D.fig'
savefig(filename);


    






    