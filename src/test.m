function [x ,y, t, p] = test(gridSize, linesrc, linedst, nSamplePerGrid, lineConstraintType,...
    deformEnergyWeights, w, h)

%dstImg = imread([mp_path, mp_dst_name]);
%srcImg = imread([mp_path, mp_src_name]);


%w1 = size(dstImg, 2);
%h2 = size(dstImg, 1);
[x,uv, p, qmSize, B, t] = buildRegularMesh(w, h, gridSize);

%update energy
L = PolyMeshEnergy(x, p, deformEnergyWeights);


% discretisice lines
[psrc, pdst] = sampleLines([linesrc, linedst], nSamplePerGrid/gridSize);


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
