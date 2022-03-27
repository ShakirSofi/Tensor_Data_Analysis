
function
% Minimize the Gabor-2D function via tensor completion
% 
%      min  f(x,y) 
%  
%
% Anh_Huy Phan
%
clear all
addpath(genpath('tensor_toolbox_2.6'))
addpath(genpath('poblano_toolbox_1.0'))
addpath(genpath('TR_functions'))

%%
% Gabor function 
% syms x y theta lambda cx cy Sigma;
% 
% % Orientation
% x_theta=(x-cx)*cos(theta)+(y-cy)*sin(theta);
% y_theta=-(x-cx)*sin(theta)+(y-cy)*cos(theta);
% 
% 
% f = exp(-1/2 * (x_theta.^2 + y_theta.^2)/Sigma^2) .* cos(2*pi/lambda * x_theta);
% 
% g = gradient(f,[x y]);

%% Parameters for Gabor2D

theta_ = pi/3;  %2*pi*rand;
lambda_  = 20;
Sigma_ = 10;

gridsize = [64, 64];

% Center of gaussian window
cx_ = 0.5*gridsize(1);
cy_ = 0.5*gridsize(2);

fxy = @(x,y) f_gabor2D([x y],theta_,Sigma_, lambda_, cx_, cy_);

%% Trust Region algorithm to find minimum of f(x,y)

% Initial point 
xy0 = [10 10];

options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display', 'iter','OptimalityTolerance',1e-10,'StepTolerance',1e-10);
xy_ = fminunc(@(xy) fxy(xy(1),xy(2)), xy0,options);

% 
fxy_ = fxy(xy_(1),xy_(2));
fprintf('Trust-region  (x,y) = (%d, %d)   f(x,y) = %d \n',xy_(1),xy_(2),fxy_);

%% Visualize Gabor-2D

fsurf(fxy,[0 gridsize(1) 0 gridsize(2)])

hold on
plot3(xy_(1),xy_(2),fxy_,'ro','linewidth',4)


%% Sample the Gabor2D function and create a matrix with missing elements
% F = [f(x_i, y_j)] 

[xd, yd] = meshgrid(1:gridsize(1), 1:gridsize(2));

% Get full matrix F (used as ground truth for comparison)
F = fxy(xd(:), yd(:));
F = reshape(F,size(xd));


% Samplying the function f(x,y)  e.g.,
P_Omega= rand(size(xd))>0.82;

% Orientation

xd(~P_Omega) = nan;
yd(~P_Omega) = nan;

Fomega = fxy(xd(:), yd(:));
Fomega = reshape(Fomega,size(P_Omega));
Fomega(~P_Omega) = 0; 


%% CPD for incomplete tensor (matrix) 

R = 2; % rank of the decomposition 
best_err = inf;
for kk = 1:20
    [P, P0, output] = cp_wopt(tensor(Fomega),tensor(P_Omega),R,'alg_options',struct('MaxIters',100));
    if output.OptOut.FuncEvals(end) <best_err
        best_err = output.OptOut.FuncEvals(end);
        Pbest = P;
    end
end

[P, P0, output] = cp_wopt(tensor(Fomega),tensor(P_Omega),R,'init',Pbest.U,'alg_options',struct('MaxIters',1000));

%% Check the approximation error 
Fhat = double(full(P));
err = norm(tensor(F - Fhat))/norm(F(:))

Fnan = Fomega;
Fnan(~P_Omega) = nan;

clf
imagesc([F nan(size(F,1),2)  Fnan  nan(size(F,1),2) double(full(P))])
title('f(x,y)   -  Sampling F   - Estimated f(x,y)')


fprintf('Fmin   estimate  %d, true   %d \n',min(Fhat(:)), min(F(:)))
fprintf('Fmax  estimate  %d, true   %d \n',max(Fhat(:)), max(F(:)))

 
%% Example 2 : find minimum of Gabor-3D function

clear all;

% Parameters for Gabor2D
theta = pi/3 ; % 2*pi*rand;
phi = 2*pi*rand;
lambda  = 20;
Sigma = 10;

gridsize = [64 64 64];


% Center of gaussian window
cx = 0.5*gridsize(1);
cy = 0.5*gridsize(2);
cz = 0.5*gridsize(3);

fxyz = @(xyz) f_gabor3D(xyz,theta,phi,Sigma, lambda, cx, cy,cz);
 
%% Trust Region algorithm to find minimum of f(x,y)
% Initial point 
xyz0 = [10 10 10];

options = optimoptions('fminunc','Display', 'iter','OptimalityTolerance',1e-10,'StepTolerance',1e-10);
 
xyz1 = fminunc(fxyz,xyz0,options )

fmin = f_gabor3D(xyz1,theta,phi,Sigma, lambda, cx, cy,cz)

fprintf('Trust-region  (x,y) = (%.2f, %.2f,%.2f)   f(x,y) = %d \n',xyz1(1),xyz1(2), xyz1(3),fmin);


%% Sample the Gabor3D function and create a matrix with missing elements
% F = [f(x_i, y_j)] 
% Generate mesh
[x, y, z] = meshgrid(1:gridsize(1), 1:gridsize(2),1:gridsize(3));

% Get full matrix F (used as ground truth for comparison)
F = fxyz([x(:), y(:) , z(:)]);
F = reshape(F,size(x));


% Samplying the function f(x,y,z)  with only 4% samples 
P_Omega= rand(size(x))>0.96; 

x(~P_Omega) = nan;
y(~P_Omega) = nan;
z(~P_Omega) = nan;

Fomega = fxyz([x(:), y(:) , z(:)]);
Fomega = reshape(Fomega,size(P_Omega));
Fomega(~P_Omega) = 0; 
Fnan = Fomega;Fnan(~P_Omega) = nan;

%% Approximation using CPD of imcomplete data

R = 10; % rank of the CPD for incomplete data 
best_err = inf;
for kk = 1:10
    [P, P0, output] = cp_wopt(tensor(Fomega),tensor(P_Omega),R,'alg_options',struct('MaxIters',100));
    if output.OptOut.FuncEvals(end) <best_err
        best_err = output.OptOut.FuncEvals(end);
        Pbest = P;
    end
end

[P, P0, output] = cp_wopt(tensor(Fomega),tensor(P_Omega),R,'init',Pbest.U,'alg_options',struct('MaxIters',2000));

%% Check relative error
Pf = full(P);

err = norm(tensor(F - Pf))/norm(F(:))


fprintf('Fmin   estimate  %d, true   %d \n',min(Pf(:)), min(F(:)))
fprintf('Fmax  estimate  %d, true   %d \n',max(Pf(:)), max(F(:)))
 
%% Approximation using Tensor Chain - Looped Tensor Network

R = [3 3 3];
best_err = inf;
for kk = 1:10
    [Y_hat,Ux,out_opt1]=WTR(Fomega, P_Omega,R,100);
    if out_opt1.TraceFunc(end) <best_err
        best_err = out_opt1.TraceFunc(end);
        Uxbest = Ux;
    end
end
[Y_hat,Ux2,out_opt2]=WTR(Fomega, P_Omega,R,3000,Uxbest);


%% Check relative error

err = norm(tensor(F - Y_hat))/norm(F(:))

%%
imagesc([squeeze(F(32,:,:))  squeeze(Fnan(32,:,:))   squeeze(Y_hat(32,:,:))])

fprintf('Fmin   estimate  %d, true   %d \n',min(Y_hat(:)), min(F(:)))
fprintf('Fmax  estimate  %d, true   %d \n',max(Y_hat(:)), max(F(:)))


%%
function [f,g] = f_gabor2D(xy,theta,Sigma, lambda, cx, cy)

x = xy(:,1);
y = xy(:,2);

% Gabor 2D
f = exp(-((cos(theta)*(cx - x) + sin(theta)*(cy - y)).^2/2 + (cos(theta)*(cy - y) - sin(theta)*(cx - x)).^2/2)/Sigma^2).*cos((2*pi*(cos(theta)*(cx - x) + sin(theta)*(cy - y)))/lambda);

if nargout > 1
    % gradient of f
    g = [(exp(-((cos(theta)*(cx - x) + sin(theta)*(cy - y))^2/2 + (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/2)/Sigma^2)*cos((2*pi*(cos(theta)*(cx - x) + sin(theta)*(cy - y)))/lambda)*(cos(theta)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(theta)*(cos(theta)*(cy - y) - sin(theta)*(cx - x))))/Sigma^2 + (2*pi*sin((2*pi*(cos(theta)*(cx - x) + sin(theta)*(cy - y)))/lambda)*exp(-((cos(theta)*(cx - x) + sin(theta)*(cy - y))^2/2 + (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/2)/Sigma^2)*cos(theta))/lambda
        (exp(-((cos(theta)*(cx - x) + sin(theta)*(cy - y))^2/2 + (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/2)/Sigma^2)*cos((2*pi*(cos(theta)*(cx - x) + sin(theta)*(cy - y)))/lambda)*(cos(theta)*(cos(theta)*(cy - y) - sin(theta)*(cx - x)) + sin(theta)*(cos(theta)*(cx - x) + sin(theta)*(cy - y))))/Sigma^2 + (2*pi*sin((2*pi*(cos(theta)*(cx - x) + sin(theta)*(cy - y)))/lambda)*exp(-((cos(theta)*(cx - x) + sin(theta)*(cy - y))^2/2 + (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/2)/Sigma^2)*sin(theta))/lambda];
end
end

%%
function [f,g] = f_gabor3D(xyz,theta,phi,Sigma, lambda, cx, cy,cz)
% f(x,y,z) : Gabor function 
% gradient of f w.r.t (x,y,z)

x = xyz(:,1);
y = xyz(:,2);
z = xyz(:,3);


% Orientation
x_theta1=(x-cx)*cos(theta)+(y-cy)*sin(theta);
y_theta=-(x-cx)*sin(theta)+(y-cy)*cos(theta);
z_theta = z;

% pitch - rotation about y
x_theta = x_theta1*cos(phi)-(z_theta - cz)*sin(phi);
%y_theta = y_theta;
z_theta = x_theta1*sin(phi)+(z_theta - cz)*cos(phi);

% Generate gabor
f = exp(-.5*(x_theta.^2/Sigma^2+y_theta.^2/Sigma^2+z_theta.^2/Sigma^2)).*cos(2*pi/lambda*x_theta);

if nargout > 1
    g = [cos((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*((cos(phi)*cos(theta)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/Sigma^2 - (sin(theta)*(cos(theta)*(cy - y) - sin(theta)*(cx - x)))/Sigma^2 + (cos(phi)*cos(theta)*sin(phi)*(sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z)))/Sigma^2) + (2*pi*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*sin((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*cos(phi)*cos(theta))/lambda
        cos((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*((cos(theta)*(cos(theta)*(cy - y) - sin(theta)*(cx - x)))/Sigma^2 + (cos(phi)*sin(theta)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/Sigma^2 + (cos(phi)*sin(phi)*sin(theta)*(sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z)))/Sigma^2) + (2*pi*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*sin((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*cos(phi)*sin(theta))/lambda
        - cos((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*((sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/Sigma^2 - ((cos(phi) - sin(phi)^2)*(sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z)))/Sigma^2) - (2*pi*exp(- (sin(phi)*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)) + cos(phi)*(cz - z))^2/(2*Sigma^2) - (cos(theta)*(cy - y) - sin(theta)*(cx - x))^2/(2*Sigma^2) - (cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z))^2/(2*Sigma^2))*sin((2*pi*(cos(phi)*(cos(theta)*(cx - x) + sin(theta)*(cy - y)) - sin(phi)*(cz - z)))/lambda)*sin(phi))/lambda];
    
end
end

end