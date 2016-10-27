function [y, energy] = deformAAAP(x, Asrc, pdst, L, flexLineConstraints)

% AAAP/ASAP deform a quadmesh with line constraints
% [y, energy] = deformAAAP(x, Asrc, pdst, L, flexLineConstraints)
% input: 
%        x: geometry of the original quadmesh
%        Asrc: matrix that express lines (sampled points on lines) as linear 
%        combinations of x
%        pdst: target positions of the lines (sampled points on them), each 
%        cell element corresponds to one line
%        L: AAAP/ASAP energy of the quadmesh
%        flexLineConstraints: constraint type of each line
% output:
%        y: geometry of the deformed quadmesh
%        energy: AAAP/ASAP energy of the deformation

fR2C = @(x) complex(x(:,1), x(:,2));

if nargin<5, flexLineConstraints = 0; end

realx = isreal(x);
if realx, x = fR2C(x); end

nv = size(x, 1);

B1 = [];
nb = numel(B1);



%% lines with sliding samples
if flexLineConstraints>0
    nSamplesInLine = cellfun(@numel, pdst);
    AIdxs = cumsum([1; nSamplesInLine]);
    nlines = numel(pdst);
    C = cell(nlines, 1);
    d = cell(nlines, 1);
    
    C2 = cell(nlines, 1);
    d2 = cell(nlines, 1);
    for i=1:nlines
        a = pdst{i}(1);
        b = pdst{i}(end);

        if flexLineConstraints==2
            A1 = Asrc(AIdxs(i):AIdxs(i+1)-1, :);
            d{i} = (imag(a)*real(b) - real(a)*imag(b))*ones(numel(pdst{i}), 1);
        else % flexLineConstraints==1
            A1 = Asrc(AIdxs(i)+1:AIdxs(i+1)-2, :);
            d{i} = (imag(a)*real(b) - real(a)*imag(b))*ones(numel(pdst{i})-2, 1);
            
            
            C2{i} = Asrc([AIdxs(i) AIdxs(i+1)-1], :);
            d2{i} = [a; b];
        end
        
        C{i} = [imag(a-b)*A1 -real(a-b)*A1];
    end
    
    
    C = cell2mat(C);
    d = cell2mat(d);
    
    if flexLineConstraints==1
        C2 = cell2mat(C2);
        d2 = cell2mat(d2);
        %% remove constraints (possibly contradicting) for same points
        [~, ia] = unique(C2*x);
        C2 = C2(ia,:);
        d2 = d2(ia);
        
        C = [C; real(C2) -imag(C2); imag(C2) real(C2)];
        d = [d; real(d2); imag(d2)];
    end
    
    
    %% remove constraints (possibly contradicting) for same points
    [d, C] = qr(C, d);
    d = d(any(C,2));
    C = C(any(C,2),:);

    Lr = [real(L) -imag(L); imag(L) real(L)];
    y = [Lr*2 C.'; C sparse(numel(d), numel(d))] \ [zeros(nv*2,1); d];
    
    %% soft constraints
%     lambda = 10;
%     y = (Lr*2+lambda*C'*C)\(lambda*C'*d);
    y = complex(y(1:nv), y(nv+(1:nv)));
else % flex_line == 0
    pdst = cell2mat(pdst);
    pdst = reshape(pdst, [], 1); % merge multiple lines

    %% remove constraints (possibly contradicting) for same points
    [~, ia] = unique(Asrc*x);
    Asrc = Asrc(ia, :);
    pdst = pdst(ia);
    
    %% solve with hard-constraints
    C = [sparse(1:nb, B1, 1, nb, nv); Asrc];
    d = [x(B1); pdst];
    [d, C] = qr(C, d);
    d = d(any(C,2));
    C = C(any(C,2),:);

    %% solve in complex
    y = [L*2 C.'; C sparse(numel(d), numel(d))] \ [zeros(nv,1); d];
    y = y(1:nv);

    %% soft constraints
%      lambda = 1e1;
%      y = (L+lambda*C'*C)\(lambda*C'*d);
end

energy = y'*L*y;

if realx,    y = [real(y) imag(y)]; end

