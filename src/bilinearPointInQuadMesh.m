function A = bilinearPointInQuadMesh(pts, X, P, qmSize)

% express points in a quad mesh as the convex combination of there 
% containing quads, using bilinear weights
% A = bilinearPointInQuadMesh(pts, X, P, qmSize)
% input: 
%        pts: points that are to be expressed as bilinear combinations of 
%        the quadmesh vertices
%        X, P: the vertices/connectivy of the quadmesh
%        qmSize: size (rows/columns of quads) of the quadmesh, that is 
%        consctructed to cover some image plane
% output:
%        A: a matrix that gives the weights for the points as combinations 
%        of the quadmesh vertices, i.e. A*X = pts


if isempty(pts)
    A = sparse(0, size(X,1));
    return;
end

if ~isreal(X), X = [real(X) imag(X)]; end

if ~isreal(pts)
    pts = reshape(pts, [], 1); % merge multiple lines
    pts = [real(pts) imag(pts)];
end

nx = size(X, 1);
npts = size(pts, 1);

%% make sure P is oriented CCW!
bbox = minmax(X');

%qij = [ceil( (pts(:,1)+eps-bbox(1,1))*(qmSize(2)-1)/range(bbox(1,:)) ) ...
%       ceil( (pts(:,2)+eps-bbox(2,1))*(qmSize(1)-1)/range(bbox(2,:)) )];
   
qij = [ceil( (pts(:,1)+eps-bbox(1,1))*(qmSize(2)-1)/...
    abs(max(bbox(1,:)) - min(bbox(1,:)))) ...
    ceil( (pts(:,2)+eps-bbox(2,1))*(qmSize(1)-1)/...
    abs(max(bbox(2,:)) - min(bbox(2,:))))];
   
q = (qij(:,1)-1)*(qmSize(1)-1) + qij(:,2);

wx = (pts(:,1) - X(P(q,1),1))./(X(P(q,2),1)-X(P(q,1),1));
wy = (pts(:,2) - X(P(q,1),2))./(X(P(q,4),2)-X(P(q,1),2));

A = sparse( repmat((1:npts)',1,4), P(q,:), [(1-wx).*(1-wy) wx.*(1-wy)  wx.*wy (1-wx).*wy ], npts, nx );
