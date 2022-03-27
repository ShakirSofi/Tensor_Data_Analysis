% Example for Kronecker factorization 
clear all;
clc;
I = 5; J = 6;
R = 3; S = 2;

A0 = randn(I,R);
B0 = randn(J,S);

Y = kron(A0,B0);

%%  Algorithm 
Yt = reshape(Y,[J I S R]);

Yt = permute(Yt,[1 3 2 4]);


Ym = reshape(Yt,J*S, I*R);


[u,s,v] = svds(Ym,1);

A = reshape(v,I,R);
B = reshape(u*s,J,S);

%%
norm(kron(A,B) - Y ,'fro')