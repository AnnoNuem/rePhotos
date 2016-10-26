function [p1, p2, p3] = sampleLines(lines, sampleRate)

% perform point sampling on line pairs
% [p1, p2, p3] = sampleLines(lines, sampleRate, adaptive)
% input: 
%        lines: pairs of lines to be sampled
%        sampleRate: how many points to sample
% output:
%        p1, p2, p3: sampled points on the line pairs
%        p2, p3 are converted to cells, each cell contain one line




assert( ~isempty(lines), 'no line defined!' );

fR2C = @(x) complex(x(:,1:2:end), x(:, 2:2:end));
linesrc = fR2C(lines);

fGenWt = @(n) [n-1:-1:0; 0:n-1]'/(n-1);


fNSampleAdaptive = @(pq) max(2, ceil( sampleRate*abs(pq(1)-pq(2)) ));
pts = arrayfun(@(i) fGenWt(fNSampleAdaptive(linesrc(i,1:2)))*reshape(linesrc(i,:), 2, []), 1:size(linesrc, 1), 'UniformOutput', false);


%pts = cat(1, pts{:});
nSampleInLines = cellfun(@(x) size(x,1), pts);
pts = cell2mat(pts');

p1 = pts(:,1);
if nargout>1
    p2 = pts(:,2);
    % for later use, to figure out how many points on each line
    p2 = mat2cell(p2, nSampleInLines);
end
if nargout>2
    p3 = pts(:,3);    
    p3 = mat2cell(p3, nSampleInLines);
end

