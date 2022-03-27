% Example for Khatrirao product factorization 
%    min |Y - khatrirao(C, D)|_F^2
%
clc; clear all;
%% generate data
I = 10; J = 15; R = 5;

A = randn(I,R);

B = randn(J,R);

Y = khatrirao(A,B);

%% Algorithm

Yt = reshape(Y,J,I,R);


for r = 1: R
    [u,s,v] = svds(Yt(:,:,r),1);
    
    C(:,r) = v;
    D(:,r) = u*s;
    
end

norm(khatrirao(C,D) - Y,'fro')
