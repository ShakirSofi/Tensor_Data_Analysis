% This example demonstrates CP-regression 
% Dataset: Gabor tensor of MNIST images
%
% Anh-Huy Phan
%
%%
%rootpath='/gpfs/data/home/a.phan/';
%addpath(fullfile(rootpath,'MNIST'))
%addpath(genpath(fullfile(rootpath,'TSB')))
addpath(genpath('tensor_toolbox_2.6'))
addpath(genpath('poblano_toolbox_1.0'))
addpath(genpath('TR_functions'))

%% clear all
digits = [0 1];

No_digits_ = 20; % number of images per digit
imageSize =[28 28];
orientationsPerScale = [8 8 8 8]; % assume number of orientations at scales are the same
numberBlocks=imageSize(1);% number of blocks after downsampling

%%
clear X
true_labels  = [];
for kd = 1:numel(digits)
    load(sprintf('mnist_gabor_no%d_1000.mat',digits(kd)));
    F = F(:,:,:,:,1:No_digits_);
    X(:,kd) = F(:);
    true_labels = [true_labels  ; kd*ones(No_digits_,1)];
end
No_digits = size(F,5);
SzF = size(F);

%No_digits_ = 1000; % number of images per digit
imageSize =[28 28];
orientationsPerScale = [8 8 8 8]; % assume number of orientations at scales are the same
numberBlocks=imageSize(1);% number of blocks after downsampling

X = reshape(X,[SzF(1)*SzF(2) SzF(3) SzF(4) SzF(5)*numel(digits)]);
% centralize X
N = ndims(X);
Xm = mean(X,N);
X = bsxfun(@minus,X,Xm);
disp('Size X=')
size(X)

y = true_labels;

%% Prob1: 
% 
%   min E||yk -  <X_k, W>||^2 + gamma ||W||_F^2
%
%   where W is rank-1 tensor
%

%% initialize W as best rank-1 approximation of X
opts = cp_fastals();
opts.init = 'nvecs';
W = cp_fastals(tensor(X),1,opts);

W = W.U(1:N-1);
X = tensor(X);
sz = size(X);

% visualize the initial features
ff0 = double(ttv(X,W,1:N-1));

disp('Size ff0=')
size(ff0)

figure('Name', 'FF0');clf
plot(ff0)

SVMModel = fitcsvm(ff0,true_labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)


%% Main algorithm
gamma = 0.10;
for kiter = 1:100
    
    for n = 1:N-1
        
        xmodes = setdiff(1:N-1,n);
        Z = double(ttv(X,W(xmodes),xmodes));
        
        % gamma = 0; W{n} = Z'\y;
        W{n} = (Z*Z'+gamma * eye(sz(n)))\(Z*y); % (Z'Z+0.1*I)^(-1)Z'y
        lambda = norm(W{n});
        W{n} = W{n}/lambda;
    end

    % evaluate error
    
    err(kiter) = norm(y - Z'*W{n}*lambda)^2;
    
    fprintf('%d   %d\n',kiter,err(kiter))
    
    if (kiter> 1) && abs(err(kiter)-err(kiter-1))<1e-3
        break
    end   
end

%% Exact Features 
ff = Z'*W{n};

% visualize the features
%hold on 
figure('Name', 'New')
plot(ff)

%% Train SVM

SVMModel = fitcsvm(ff,true_labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)
 

%%{
%% PART 2:  Higher Rank for Weight tensor
% 
%   min E||yk -  <X_k, W>||^2 + gamma ||W||_F^2
%
%   where W is rank-R tensor
%

rankW = 2;
% initialize W
opts = cp_fastals();
opts.init = 'nvecs';
W = cp_fastals(tensor(X),rankW,opts);

lambda = W.lambda;
W = W.U(1:N-1);
X = tensor(X);
sz = size(X);

% exact initial features
ff0 = 0;
for r = 1:rankW
    Wr = cellfun(@(x) x(:,r),W,'uni',0);
    
    ff0 = ff0+lambda(r) * double(ttv(X,Wr,1:N-1));
end
% visualize features
figure(1);clf
plot(ff0)

SVMModel = fitcsvm(ff0,true_labels,'Standardize',true,'KernelFunction','poly',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)


%% Main algorithm
gamma = 0.0;
maxiters = 100;
for kiter = 1:maxiters
    
    for n = 1:N-1
        
        xmodes = setdiff(1:N-1,n);
        Z = zeros([sz([n N]),rankW]);
        for r = 1:rankW
            Wr = cellfun(@(x) x(:,r),W,'uni',0);
            Z(:,:,r) = double(ttv(X,Wr(xmodes),xmodes));
        end
        Z = double(tenmat(Z,2)');
        % gamma = 0; W{n} = Z'\y;
        Wn= (Z*Z'+gamma * eye(size(Z,1)))\(Z*y);
        Wn = reshape(Wn,sz(n),rankW);
        
        % normalization
        lambda = sqrt(sum(Wn.^2,1));
        Wn = Wn*diag(1./lambda);
        W{n} = Wn;
    end

    % evaluate error    
    err(kiter) = norm(y - Z'*reshape(W{n}*diag(lambda),[],1))^2;
    
    fprintf('%d   %d\n',kiter,err(kiter))
    
    if (kiter> 1) && abs(err(kiter)-err(kiter-1))< (1e-5*err(kiter-1))
        break
    end   
end

%% Exact Features 
figure(1); clf
ff = Z'*reshape(W{n}*diag(lambda),[],1);
hold on 
plot(ff)

%% SVM

SVMModel = fitcsvm(ff,true_labels,'Standardize',true,'KernelFunction','poly',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)
%%}
 