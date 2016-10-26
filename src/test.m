function [x ,y, p] = test(x, p, qmSize, linesrc, linedst, nSamplePerGrid,...
    lineConstraintType, deformEnergyWeights, gridSize)

qmSize = double(cell2mat(qmSize));

%dstImg = imread([mp_path, mp_dst_name]);
%srcImg = imread([mp_path, mp_src_name]);


%w1 = size(dstImg, 2);
%h2 = size(dstImg, 1);

%update energy
% DONE
L = PolyMeshEnergy(x, p, deformEnergyWeights);


% discretisice lines
%DONE
[psrc, pdst] = sampleLines([linesrc, linedst], nSamplePerGrid/gridSize);


% WORKING
Asrc = bilinearPointInQuadMesh(psrc, x, p, qmSize);
% line constraint type 
% 0 = linear sampled
% 1 = semi flexible
% 2 = flexible
[y, energy] = deformAAAP(x, Asrc, pdst, L, lineConstraintType);
%srcLineSamples = Asrc * x;
%dstLineSamples = fC2R(cell2mat(pdst));
%mappedLineSamples = Asrc * y;

%scatter(x(:,1),x(:,2),'.');
%scatter(y(:,1),y(:,2),'.');
