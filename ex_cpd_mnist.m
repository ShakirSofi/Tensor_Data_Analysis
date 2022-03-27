%  Feature extraction 
% MNIST dataset

rootpath='/gpfs/data/home/a.phan/';
addpath(fullfile(rootpath,'MNIST'))

addpath(genpath(fullfile(rootpath,'TSB')))

addpath(genpath(fullfile(rootpath,'/tensorlab_0507')))
    

%% clear all
digits = [0 1];

No_digits_ = 1000; % number of images per digit
imageSize =[28 28];
orientationsPerScale = [8 8 8 8]; % assume number of orientations at scales are the same
numberBlocks=imageSize(1);% number of blocks after downsampling

%%
clear Y
for kd = 1:numel(digits)
    load(sprintf('mnist_gabor_no%d_%d.mat',digits(kd),No_digits_));
    F = F(:,:,:,:,1:100);
    Y(:,kd) = F(:);
end
No_digits = size(F,5);
SzF = size(F);

No_digits_ = 1000; % number of images per digit
imageSize =[28 28];
orientationsPerScale = [8 8 8 8]; % assume number of orientations at scales are the same
numberBlocks=imageSize(1);% number of blocks after downsampling

Y = reshape(Y,[SzF(1)*SzF(2) SzF(3) SzF(4) SzF(5)*numel(digits)]);

%%  CPD

opts = cp_fastals();
opts.printitn = 1;
opts.maxiters  = 1000;
R = 3; % rank of the decomposition = Number of extracted features 
[Ycp,outcp] = cp_fastals(tensor(Y),R,opts);

% %%
% Ucp =  Ycp.U;
% Ucp{1} = Ucp{1}*diag(Ycp.lambda);
% Rcp = R;

%% Visualize the extracted figures
figure(1); 
% Select the feature 1
sel_comp = 1;
plot(Ycp.U{end}(:,sel_comp))

% 
figure(2);
imagesc(reshape(Ycp.U{1}(:,sel_comp),SzF(1),[]))

figure(3);
plot(reshape(Ycp.U{2}(:,sel_comp),SzF(3),[]))

%% KMEANs
rmpath(genpath(fullfile(rootpath,'/tensorlab_0507'))) % conflict with Matlab Kmeans

%% Clustering
Nclasses = numel(digits);
gnd = [zeros(No_digits,1) ; ones(No_digits,1)];

%Features = double(Ycp.U{end})*diag(Ycp.lambda);
Features = double(Ycp.U{end}(:,sel_comp));
% [Features,s,v] = svds(Features,Nclasses);
% Features = Features *s;

[idx,ctrs] = kmeans(Features,Nclasses,'Distance','sqeuclidean',...
    'Replicates',500);
idx0 = idx;
idx = bestMap(gnd,idx);
%============= evaluate AC: accuracy ==============
acc  = length(find(gnd == idx))/length(gnd);
%============= evaluate MIhat: nomalized mutual information =================
MIhat = MutualInfo2(gnd,idx);

fprintf('Accuracy %.2f\n',acc)
fprintf('Mutual information %.2f\n',MIhat)

