function  
%
clear all
%%{
%cd('/gpfs/data/gpfs0/LinReg/SupportPackages_for_TSB');
addpath(genpath('tensor_toolbox_2.6'))
addpath(genpath('poblano_toolbox_1.0'))
addpath(genpath('TR_functions'))
%addpath(genpath('TSB/2021'))
%%}
%%

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
 
%% Tensor of the Gabor3D function 
% F = [f(x_i, y_j)] 
% Generate mesh
[x, y, z] = meshgrid(1:gridsize(1), 1:gridsize(2),1:gridsize(3));

% Get full matrix F (used as ground truth for comparison)
F = fxyz([x(:), y(:) , z(:)]);
F = reshape(F,size(x));


%%
Pc = cell(12,1);
for R = 1:12   % rank of the CPD 
    
    option = cp_fastals();
    option.init = 'nvecs'; %{'nvecs' 'fiber' 'random' 'random'};
    option.maxiters = 2000;
    option.printitn = 1;
    option.tol = 1e-12;
    option.linesearch = false;
    
    [P, output] = cp_fastals(tensor(F),R,option);

    %err(R) = real(output.Fit(end));
    err(R) = norm(F - full(P))/norm(F(:));
    ss(R) = cp_sensitivity(P);
    P_cp{R} = P;
    
end

%% Approximation error 
% The Gabor-3d tensor has rank-3
figure(1);
loglog(err)
xlabel('Rank')
ylabel('Approximation Error')


%%
% R = 3;
% Pb = P_cp{R};

% %%
% while 1
%     while 1
%         [Pb,outputb_] = exec_cp_bals(tensor(F),Pb,1e8,2);
%         if mean(abs(diff(outputb_.cost(end-10:end)))) <5e-5
%             break
%         end
%     end
%     
%     %%
%     option.init  = Pb;
%     [Pb, output] = cp_fastals(tensor(F),R,option);
% end

% intensity=100;
% option.init  = Pb;
% [Pb, output] = cp_boundlda_als(tensor(F),R,intensity,option);
% Pb = normalize(Pb);

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