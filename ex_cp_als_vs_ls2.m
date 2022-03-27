% This demo compares intialization methods using in CP algorithms.
%
% TENSORBOX, 2013
% Phan Anh Huy,  phan@brain.riken.jp
%
%{
addpath(genpath('/beegfs/home/a.phan/For courses/TSB/2021'))
addpath(genpath('/beegfs/home/a.phan/For courses/tensorlab_0507'))
%}
%%
clear all
warning off
N = 3;   % tensor order
R = 10;   % tensor rank
I = ones(1,N) * R; % tensor size


% Generate a Kruskal tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
% c = [0 0.3; 0.1 0.4; 0.5 0.99 ; 0.9 0.99; 0.9 0.999;0.9 0.99; 0.9 0.999];
c = [0.9 0.999;  0.9 0.999;  0.9 0.999;  0.9 0.999; 0.9 0.999;0.9 0.999];

A = cell(N,1);
for n = 1:N
    A{n} = gen_matrix(I(n),R,c(n,:));
end

% Add Gaussian noise
Y = ktensor(A(:));  % a Kruskal tensor of rank-R
SNR = 100; % Noise level in dB, set SNR = inf for noiseless tensor
normY = norm(Y);
if ~isinf(SNR)
    Y = full(Y);
    sig2 = normY./10.^(SNR/20)/sqrt(prod(I));
    Y = Y + sig2 * randn(I);
end
Y = tensor(Y);
 

% Initialization for the CPD 
U0 = cp_init(tensor(Y),R,struct('init','dtld'));

%% Parameters for FastALS
opts = cp_fastals;
opts.linesearch = false;
opts.printitn = 1;
opts.tol = 1e-10;
opts.maxiters= 5000;

opts.init = U0;
opts.linesearch  = 0;
[P,output] = cp_fastals(Y,R,opts);

msae = SAE(A,P.U);
fit = real(output.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \n ',-10*log10(msae),fit);

%% LS

opts.init = U0;
opts.linesearch  = true;%% Visualize and Compare results

[P2,output2] = cp_fastals(Y,R,opts);

msae = SAE(A,P2.U);
fit = real(output2.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \n ',-10*log10(msae),fit);

%% ELS : Exact Line Search 

opts.LineSearch = @cpd_els;
opts.Display = 1;
opts.MaxIter = 1000;
[U,output3] = cpd_als(double(Y),U0,opts);

msae = SAE(A,U);
fit = sqrt(2*real(output3.fval(end)))/norm(Y);

fprintf('Mean squared angular error %.3f dB \nFit %.4f \n ',-10*log10(msae),fit);

Err_els = sqrt(2*output3.fval)/norm(Y)

%% Visualize and Compare results

figure(1);clf; set(gca,'fontsize',16);hold on
clrorder = get(gca,'colorOrder');
h1= plot(output.Fit(:,1),1-real(output.Fit(:,2)),'color',clrorder(1,:));
hold on
h2= plot(output2.Fit(:,1),1-real(output2.Fit(:,2)),'color',clrorder(2,:));

h3= plot(Err_els,'color',clrorder(3,:));

grid  on
h = [h1 h2 h3];
set(h,'linewidth',2)
set(gca,'yscale','log')
set(gca,'xscale','log')

xlabel('No. Iterations')
ylabel('Relative Error')    
axis tight
legend(h,{'ALS' 'ALS+LS' 'ALS+ELS'});


 return
 
 
%% ALS again

opts = cp_fastals;
opts.linesearch = false;
opts.printitn = 1;
opts.tol = 1e-10;
opts.maxiters= 100;
opts.init = U0;
opts.linesearch  = 0;
[P,output_x] = cp_fastals(Y,R,opts);

msae = SAE(A,P.U);
fit = real(output_x.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \n ',-10*log10(msae),fit);

Pb = P;
for kr =1:1
    while 1
        [Pb,outputb_] = exec_cp_bals(tensor(Y),Pb,1e8,2,@cp_anc,1);
        if mean(abs(diff(outputb_.cost(end-10:end)))) <5e-5
            break
        end
    end
    
    %%
       while 1
        [Pb,outputb_] = exec_cp_bals(tensor(Y),Pb,1e8,2,@cp_anc,2);
        if mean(abs(diff(outputb_.cost(end-10:end)))) <5e-5
            break
        end
    end
    Pss = Pb;
    %%
    opts = cp_fastals;
    opts.linesearch = false;
    opts.printitn = 1;
    opts.tol = 1e-10;
    opts.maxiters= 5000;
    opts.init  = Pb;
    [Pb, output] = cp_fastals(tensor(Y),R,opts);
end

%%
figure(1);
Fitx = [output_x.Fit; output.Fit];

h4= plot(1-real(Fitx(:,2)),'color',clrorder(4,:));


grid  on
h = [h1 h2 h3 h4];
set(h,'linewidth',2)
set(gca,'yscale','log')
set(gca,'xscale','log')

xlabel('No. Iterations')
ylabel('Relative Error')    
axis tight
legend(h,{'ALS' 'ALS+LS' 'ALS+ELS' 'SSC'});

 
% TENSORBOX v1. 2013