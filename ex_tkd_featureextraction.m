% This example demonstrates feature extraction using 
% Tucker decomposition 
%
% Dataset: Gabor tensor of MNIST images
%
% Anh-Huy Phan

%rootpath='/gpfs/data/home/a.phan/';
%addpath(fullfile(rootpath,'MNIST'))
%addpath(genpath(fullfile(rootpath,'TSB')))    


addpath(genpath('tensor_toolbox_2.6'))
addpath(genpath('poblano_toolbox_1.0'))
addpath(genpath('TR_functions'))

%% clear all
digits = [0 1];

No_digits_ = 100; % number of images per digit

clear Y
true_labels  = [];
for kd = 1:numel(digits)
    load(sprintf('mnist_gabor_no%d_%d.mat',digits(kd),No_digits_));
    F = F(:,:,:,:,1:No_digits_);
    Y(:,kd) = F(:);
    
    true_labels = [true_labels  ; kd*ones(No_digits_,1)];

end
No_digits = size(F,5);
SzF = size(F);

No_digits_ = 100; % number of images per digit
imageSize =[28 28];
orientationsPerScale = [8 8 8 8]; % assume number of orientations at scales are the same
numberBlocks=imageSize(1);% number of blocks after downsampling

Y = reshape(Y,[SzF(1)*SzF(2) SzF(3) SzF(4) SzF(5)*numel(digits)]);

%% Tucker decomposition
Ym = mean(Y,4);
R = [2 2 2];
U0 = {nvecs(tensor(Ym),1,R(1)) nvecs(tensor(Ym),2,R(2)) nvecs(tensor(Ym),3,R(3))};

T = mtucker_als(tensor(Y),R,struct('dimorder',1:3),U0);

%% Extract features
F = double(tenmat(T.core,4));
F = F(:,3:4); % first feature often shows the mean component


SVMModel = fitcsvm(F,true_labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)


%%
sv = SVMModel.SupportVectors;
figure(1);clf
gscatter(F(:,1),F(:,2),true_labels)
hold on
plot(sv(:,2),sv(:,1),'ko','MarkerSize',10)
axis auto

legend(sprintf('digit%d',digits(1)),sprintf('digit%d',digits(2)),'Support Vector')
hold off
 