% orthogonal procrustes problem
% Anh-Huy Phan

clear all;

% Y - X * A 
I = 5; J = 8;T = 100;
R = 2; S = 5;

C0 = orth(randn(I,R));
D0 = orth(randn(J,S));
A = randn(R*S,T);

Y = kron(C0,D0)*A;
% Y = Y + .0001*randn(size(Y));

% 
% %imfile = 'lena.bmp';
% imfile = 'pepper2.tiff';
% s1rgb = imread(imfile);
% [pp,imname,ex] = fileparts(imfile);
% Y = im2double(s1rgb) ;
% Y = rgb2gray(Y);
% 
% T = size(Y,2);
% I = 32; J = size(Y,1)/I;
% R = 20; S = 20;
% A = Y(randperm(size(Y,1),R*S),:);

%% Initialize C
C0 = orth(randn(I,R));

%%
C = C0;
err = [];
for krun = 1:1000
    
    %% Update D

    Yc = kron(C',eye(J))*Y;
    Yc = reshape(Yc,J,R,[]);
    
    Yc = permute(Yc,[1 3 2 ]);
    Yc = reshape(Yc,J,[]);
    
    Ac = reshape(A,S,R,[]);
    Ac = permute(Ac,[1 3 2 ]);
    Ac = reshape(Ac,S,[]);
    
    [u,s,v] = svd(Yc*Ac','econ');
    D = u*v';
    
    err  = [err norm(Y - kron(C,D)* A,'fro')];
    
    %% Update C
    Yd = kron(eye(I),D')*Y;
    
    P_IS = per_vectrans(I,S);
    P_RS = per_vectrans(R,S);
    
    % kron(C,eye(S)) - P_IS * kron(eye(S),C) * P_RS'
    
    Ad = P_RS'*A;
    Yd = P_IS'*Yd;
    
    Yd = reshape(Yd,I,S,[]);
    
    Yd = permute(Yd,[1 3 2 ]);
    Yd = reshape(Yd,I,[]);
    
    Ad = reshape(Ad,R,S,[]);
    Ad = permute(Ad,[1 3 2 ]);
    Ad = reshape(Ad,R,[]);
    
    [u,s,v] = svd(Yd*Ad','econ');
    C = u*v';
    
    err  = [err norm(Y - kron(C,D)* A,'fro')];
    fprintf('krun % d  Error %d \n', krun, err(end))
    if abs(err(end) - err(end-1))< 1e-9
        break
    end
end
 

%%

Yt = reshape(Y,J,I,[]);
At = reshape(A,S,R,[]);
C = C0;
%%
err = [];
for krun = 1:1000
    
    %% Update D
    Yc = ttm(tensor(Yt),C',2);
    Yc = double(tenmat(Yc,1));
    Ac = double(tenmat(At,1));
    
    [u,s,v] = svd(Yc*Ac','econ');
    D = u*v';
    
    err  = [err norm(Y - kron(C,D)* A,'fro')];
    
    %% Update C
    Yd = ttm(tensor(Yt),D',1);
    Yd = double(tenmat(Yd,2));
    Ad = double(tenmat(At,2));
    
    [u,s,v] = svd(Yd*Ad','econ');
    C = u*v';
    
    err  = [err norm(Y - kron(C,D)* A,'fro')];
    fprintf('krun % d  Error %d \n', krun, err(end))
    if abs(err(end) - err(end-1))< 1e-9
        break
    end
end
 

return 

 
%% Stiefel manifold
addpath(genpath('C:\Users\huyph\Documents\MATLAB\Manifold\Manopt_6.0\manopt'))

X_m = test_wprc(Y,A,W);

 
%% Compare two solutions

fval_rm =  norm(W.*(Y-A*X_m'),'fro')^2;
fval_admm =  norm(W.*(Y-A*X),'fro')^2;

fprintf('ADMM approx_err = %d \n',fval_admm)
fprintf('Manopt approx_err = %d \n',fval_rm)

%%
function [X,fval,err,T] = main_admm_wpr(Y,A,W,X0,gamma,T)

X = X0;
%T = zeros(size(Y));
Yx = A*X;Yw = Y.*W;
for ki = 1:1000
    % Z
    Z = (Yw + gamma * (Yx + T))./(W+gamma);

    % X 
    [u,s,v] = svd(A'*Z,'econ');
    X = u*v';
    Yx = A*X;

    % T
    T = T + Yx-Z;

    fval(ki) = norm(W.*(Y - Yx),'fro');
    err(ki) = norm(Z - Yx,'fro');
    fprintf('%d %d \n',fval(ki),err(ki))

    if err(ki)< 1e-5
        break
    end
end
end


%%

%%
function X = test_wprc(Y,A,W)
% Anh- Huy Phan
M = stiefelfactory(size(Y,2),size(A,2));

% Setup the problem structure with manifold M and cost+grad functions.
problem.M = M;
problem.cost = @(X) cost(X,Y,A,W);
problem.grad = @(X) grad(X,Y,A,W);
 

%% For debugging, it's always nice to check the gradient a few times.
% checkgradient(problem);
    
%% Call a solver on our problem. This can probably be much improved if a
% clever initial guess is used instead of a random one.
X = steepestdescent(problem);

function [f] = cost(X,Y,A,W)
f = 1/2*norm(W.*(Y - A * X'),'fro')^2;
end

% Riemannian gradient of the cost function.
function [g] = grad(X,Y,A,W)

Yx = A*X';

egrad = (W.*(Yx-Y))'*A;
% then transform this Euclidean gradient into the Riemannian
% gradient.
g = M.egrad2rgrad(X, egrad);
end

end