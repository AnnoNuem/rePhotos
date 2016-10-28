function L = PolyMeshEnergy(x, p, wts)

% construct the quadratic energy matrix for AAAP defomration of a quadmesh
% L = PolyMeshEnergy(x, p, wts)
% input: 
%        x, p: the vertices and connectivity of the quadmesh
%        wts:  corresponds to [alpha beta] in the following paper
%              Generalized As-Similar-As-Possible Warping with Applications in Digital Photography
%              beta/alpha controls how much similarity is favored in the result
% output:
%        L: the matrix corresponds to the quadratic ASAP/AAAP energy


nv = size(x, 1);

n = size(p, 1);
Ais = cell(n, 1);
Ajs = cell(n, 1);
As = cell(n, 1);


A1 = [1 -1 1 -1; -1 1 -1 1; 1 -1 1 -1; -1 1 -1 1]/4;
% make sure quad is oriented in CCW order
A2 = [1 1i -1 -1i; -1i 1 1i -1; -1 -1i 1 1i; 1i -1 -1i 1]/4;
A = A1*wts(1) + A2*wts(2);

A = A*2;

%%
for i=1:n
    vvi = p(i,:); 
    nvv = numel(vvi);

    vvii = repmat(vvi, nvv, 1);
    Ais{i} = reshape(vvii', [], 1);
    Ajs{i} = reshape(vvii, [], 1);
    As{i} = A(:);
end

L = sparse(cell2mat(Ais), cell2mat(Ajs), cell2mat(As), nv, nv);

