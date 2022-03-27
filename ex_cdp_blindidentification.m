addpath(genpath('/beegfs/home/a.phan/tensorlab_0507'))
addpath(genpath('/beegfs/home/a.phan/tensor_toolbox_2.6'))
addpath(genpath('/beegfs/home/a.phan/For courses/TSB/2021'))
addpath(genpath('/beegfs/home/a.phan/For courses/CAF'))

%% Generate source signals
clear all;
nosources = 3;
nosamples = 10000;
nosensors = 3;

S = double(randn(nosources,nosamples)>0);

% Mixing matrix 
A = randn(nosensors,size(S,1));

% Generate Mixtures
X = A * S; 


%% Add Gaussian noise into the tensor Y
Noise_ = randn(size(Y0));
Noise_ = Noise_/std(Noise_(:));
Y=Y0+sigma_noise(ksnr)*Noise_;

%% Method 1: decompose Cummulant tensor
segment_length = size(S,2)-100;
clear Cx Cs;
for ki = 1:10
    ix = randi(size(X,2)-segment_length+1,1);

    [Cx(:,:,:,:,ki),M4(:,:,:,:,ki),Cx2(:,:,ki)] = cum4(X(:,ix:ix+segment_length-1)',0);
    Cs(:,:,:,:,ki) = cum4(S(:,ix:ix+segment_length-1)',0);
end

% 
% norm(Cx - ttm(tensor(Cs),{A A A},[1 2 3]))


%% CPD of the Cummulant tensor to get the mixing matrix H
options = cp_fastals();

options.init = {'nvecs' 'rand', 'fiber' 'dtld'};
options.tol = 1e-12;
options.maxiters = 1000;
options.printitn = 1;
options.linesearch = 1;
R  = nosources;

P = cp_fastals(tensor(Cx),R,options); 

%%  Get estimated mixing matrix A
Ah = P.U{1}; 

% Retrieve the sources and assess performance 
Shat = Ah\X;

Shat = diag(1./max(abs(Shat),[],2)) * Shat;

% Performance assessment 
[msae,msae2,sae,sae2,src_reorder] = SAE({Shat'},{S'});


fprintf('SAE  %s  \n',  sprintf('%.2f  dB   ', -10*log10(sae)))

%%
% %%
% Cx2 = ttm(tensor(Cx),{pinv(P.U{2})  pinv(P.U{2})},[1 2]);
% options.init = {'nvecs' 'rand', 'fiber' 'dtld'};
% P2 = cp_fastals(tensor(Cx2),R,options);
% 
% %%
% options.init = P2;
% P2 = cp_fastals(tensor(Cx2),R,options);
% 
% %%
% Ah = P.U{1}*P2.U{1};
% Shat = Ah\X;
% 
% % Shat = diag(1./max(abs(Shat),[],2)) * Shat;
% 
% % Performance assessment 
% [msae,msae2,sae,sae2,src_reorder] = SAE({Shat'},{S'});
% sae

%% Method 2 :  Decompose Characterstic tensor
d = 2; % Order of CAF
nosamplingvectors = 200;
Tx = zeros([size(X,1)^d, nosamplingvectors]);
for  ki = 1:nosamplingvectors
    u = randn(size(X,1),1);
    Tk = gen_caf(u,X,d);
    Tx(:,ki) = Tk(:);
end
Tx = reshape(Tx,[size(Tk),nosamplingvectors]);

options = cp_fastals();
options.init = {'nvecs' 'rand', 'fiber' 'dtld'};
options.printitn = 1;
R  = size(A,2);

P = cp_fastals(tensor(Tx),R,options);

% Reconstruct the sources
Shat = P.U{1}\X;

Shat = diag(1./max(abs(Shat),[],2)) * Shat;
% Performance assessment 
[msae,msae2,sae,sae2,src_reorder] = SAE({Shat'},{S'});

fprintf('SAE  %s  \n',  sprintf('%.2f  dB   ', -10*log10(sae)))

 
return
%%
% clf
% plot(Shat(1,:),S(src_reorder(1),:))
% 
% hold on
% plot(Shat(2,:),S(src_reorder(2),:))
% 
% 
% plot(-Shat(3,:),S(src_reorder(3),:))


%%
% 
% %%
% Pb = P;
% for kr =1:2
%     while 1
%         [Pb,outputb_] = exec_cp_bals(tensor(Cx),Pb,1e8,2,@cp_anc,1);
%         if mean(abs(diff(outputb_.cost(end-10:end)))) <5e-5
%             break
%         end
%     end
%     
%     %%
%        while 1
%         [Pb,outputb_] = exec_cp_bals(tensor(Cx),Pb,1e8,2,@cp_anc,2);
%         if mean(abs(diff(outputb_.cost(end-10:end)))) <5e-5
%             break
%         end
%     end
%     
%     %%
%     options.linesearch = 0;
%     options.init  = Pb;
%     [Pb, output] = cp_fastals(tensor(Cx),R,options);
% end
% % 
% % intensity=100;
% % option.init  = Pb;
% % [Pb, output] = cp_boundlda_als(tensor(Cx),R,intensity,option);
% % Pb = normalize(Pb);
% %%
% 
% Shat = Pb.U{1}\X;
% 
% Shat = diag(1./max(abs(Shat),[],2)) * Shat;
% 
% % Performance assessment 
% [msae,msae2,sae,sae2,src_reorder] = SAE({Shat'},{S'});
% sae
% % 
% % %%
% 
% 
% function  c2 = cum2(X,prewhiten)
% % Center the variables.
% X = bsxfun(@minus,X,mean(X,1));
% 
% % Apply a prewhitening to X if requested.
% n = size(X,1);
% if prewhiten
%     [U,S,~] = svd(X,'econ');
%     X = U*(S*pinv(S))*sqrt(n);
% end