% Example for Kronecker approximation of an image
% Y = sum_p   A_p kron B_p

clear all;
clc

%imfile = 'lena.bmp';
imfile = 'fruit2dct.tif';
s1rgb = imread(imfile);
[pp,imname,ex] = fileparts(imfile);
Y = im2double(s1rgb) ;
Y = rgb2gray(Y);

T = size(Y,2);
I = 64; R = 64; % size of A_p
J = 8; S = 8; % size of B_p
P = 30; % the number of KRON terms

%% convert image to matrix of lowrank

Yt = reshape(Y,[J I S R]);

Yt = permute(Yt,[1 3 2 4]);


Ym = reshape(Yt,J*S, I*R);


%% Find A1 kron B1 + A2 kron B2 + ...

[u,s,v] = svds(Ym,P);

A = reshape(v,I,R,P);
B = reshape(u*s,J,S,P);

%% build the approximation 

Yhat = 0;
for p  =1:P
    Yhat = Yhat + kron(A(:,:,p),B(:,:,p));
end

%%
clf
imagesc([Y nan(size(Y,1),5) Yhat])