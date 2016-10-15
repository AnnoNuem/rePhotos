function A = mlsDeformLinesMatrix(linesrc, x, energy)

if nargin<3, energy = 'similarity'; end
if strcmp(energy, 'affine')
    assert(size(linesrc,1)>1, 'at least 2 lines are needed');
end

% fRotate = @(x) [-x(:,2) x(:,1)];
fRotate = @(x) x*complex(0, 1);
fR2C = @(x) complex(x(:,1), x(:,2));
fC2R = @(x) [real(x) imag(x)];
fdot = @(x, y) real(x).*real(y) + imag(x).*imag(y);

%%
a = fR2C(linesrc(:, 1:2));
b = fR2C(linesrc(:, 3:4));
v = fR2C(x);

% mapping between lines: ab <-> cd
nlines = size(a, 1);
nv = size(v, 1);

%%
delta00 = zeros(nv, nlines);
delta01 = zeros(nv, nlines);
delta11 = zeros(nv, nlines);

for i=1:nlines
    delta = fdot( fRotate(a(i) - v), a(i) - b(i) );
    
    theta = atan( fdot(b(i)-v, b(i)-a(i))./fdot(fRotate(b(i)-v), b(i)-a(i)) ) ...
           -atan( fdot(a(i)-v, a(i)-b(i))./fdot(fRotate(a(i)-v), a(i)-b(i)) );
       
    beta00 = fdot( a(i)-v, a(i)-v );
    beta01 = fdot( a(i)-v, v-b(i) );
    beta11 = fdot( v-b(i), v-b(i) );

    cc = abs(a(i) - b(i))/2./delta.^2;
    delta00(:, i) = cc .* (beta01./beta00 - beta11.*theta./delta);
    delta01(:, i) = cc .* (             1 - beta01.*theta./delta);
    delta11(:, i) = cc .* (beta01./beta11 - beta00.*theta./delta);
end
    
%%
% m1 = delta00 + delta01;
% m2 = delta01 + delta11;
% den = sum([m1 m2], 2);
% pstar = (m1*a + m2*b)./den;

m = [ delta00 + delta01 delta01 + delta11 ];
m = m ./ repmat(sum(m, 2), 1, size(m,2));
pstar = m*[a; b];


%%
A = zeros(nv, 2*nlines);
for i=1:nv
    %% Affine
    if strcmp(energy, 'affine')
        D = zeros(2);
    %     tmp2 = zeros(2);
        for j=1:nlines
            w = [delta00(i, j) delta01(i, j); delta01(i, j) delta11(i, j)];

            abhat = fC2R([a(j); b(j)] - pstar(i));
            D = D + abhat'*w*abhat;
    %         cdhat = fC2R([c(j); d(j)] - qstar(i));
    %         tmp2 = tmp2 + abhat'*w*cdhat;
        end

    %     M{i} = D\tmp2;

        for j=1:nlines
            w = [delta00(i, j) delta01(i, j); delta01(i, j) delta11(i, j)];
            abhat = fC2R([a(j); b(j)] - pstar(i));
            A(i, j*2+(-1:0)) = fC2R(v(i) - pstar(i))*(D\abhat'*w);
        end
    elseif strcmp(energy, 'similarity')
        %% Similarity
        miu =    delta00(i,:)*abs(a - pstar(i)).^2 ...
             + 2*delta01(i,:)*fdot(a - pstar(i), b - pstar(i)) ...
             +   delta11(i,:)*abs(b - pstar(i)).^2;

        for j=1:nlines
            w = [delta00(i, j) delta01(i, j); delta01(i, j) delta11(i, j)];
%             fHelp = @(x) fC2R( [x-pstar(i); -fRotate(x-pstar(i))] );
%             B = ( kron(w, eye(2))*[fHelp(a(j)); fHelp(b(j))]*fHelp(v(i))'/miu )';
            A(i, 2*j+(-1:0)) = ( w*conj([a(j); b(j)]-pstar(i))*(v(i)-pstar(i))/miu ).';
        end
    else
        assert(false, 'undefined energy');
    end
end

% c = fR2C(linedst(:, 1:2));
% d = fR2C(linedst(:, 3:4));
% qstar = fC2R( m*[c; d] );
% % qstar = fC2R( (m1*c + m2*d)./den );
% y = fC2R( A*reshape([c d].', [], 1) ) + qstar;

A = A + m*sparse(1:nlines*2, [1:2:nlines*2 2:2:nlines*2], 1);

%%
% m = m*sparse(1:nlines*2, [1:2:nlines*2 2:2:nlines*2], 1);
