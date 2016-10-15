function [x, uv, p, qmSize, B, t] = buildRegularMesh(w, h, gridsize)

% construct a quadmesh that covers the image plane
% [x, uv, p, qmSize, B, t] = buildRegularMesh(w, h, gridsize)
% input: 
%        w, h: width height of the image
%        gridsize: size of the squares in the output quadmesh
% output:
%        x, uv: vertex coordinates/uv of the quad mesh
%        p: quads, connectivity
%        qmSize: size of (rows/columns of squares) the quadmesh
%        B: indices of vertices on the boundary of the quadmesh
%        t: a triangulation of the quadmesh

m = ceil(h/gridsize)+1;
n = ceil(w/gridsize)+1;

% m = ceil((h-1)/gridsize+1); n = ceil(1+(w-1)/gridsize);
[x, y] = meshgrid(0:n-1, 0:m-1);
x =  [x(:) y(:)];
x = x*gridsize;


% bugfixed: updated the ordering of p in CCW, it affected inverse bilinear mapping (opengl)
p = [ reshape( repmat( (1:m-1)', 1, n-1) + m*repmat(0:n-2, m-1, 1), [], 1 ) ...
      reshape( repmat( (1:m-1)', 1, n-1) + m*repmat(1:n-1, m-1, 1), [], 1 ) ...
      reshape( repmat( (2:m)',   1, n-1) + m*repmat(1:n-1, m-1, 1), [], 1 ) ...
      reshape( repmat( (2:m)',   1, n-1) + m*repmat(0:n-2, m-1, 1), [], 1 ) ];


t = [p(:, [1 2 3]); p(:, [3 4 1])];


uv = bsxfun(@rdivide, x, [w h]);
qmSize = [m n];
B = [1 m m*(n-1)+1 m*n];
