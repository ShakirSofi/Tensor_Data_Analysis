clc;clear all
A=(imread('peppers.bmp'));

A=double(im2gray(A));
S_X=size(A);

M=80;
N=80;
S_1=randsample(S_X(1),M);
S_2=randsample(S_X(2),N);
C=A(:,S_1);
R=A(S_2,:);
X=pinv(C)*A;
subplot(1,5,1)
imshow(uint8(C*X))

subplot(1,5,2)
S_2=randsample(S_X(2),N);
R=A(S_2,:);
X=A*pinv(R);
imshow(uint8(X*R))

subplot(1,5,3)
U=pinv(A(S_2,S_1));
imshow(uint8(C*U*R))

subplot(1,5,4)
U=pinv(C)*A*pinv(R);
imshow(uint8(C*U*R))

subplot(1,5,5)
[u,s,v]=svds(double(A),30);
imshow(uint8(u*s*v'))

%%%%%%%%%%%%% CX Approximation
A=randn(500,5)*randn(5,500);
nA=norm(A);
S_X=size(A);
Er=[];
for i=1:1000
M=5;
N=5;
S_1=randsample(S_X(1),M);
C=A(:,S_1);
X=pinv(C)*A;
Er=[Er,norm(A-C*X,'fro')/nA];
end
semilogy((Er),'*')
%%%%%%%%%%%%%   RX Approximation
A=randn(500,5)*randn(5,500);
nA=norm(A);
S_X=size(A);
Er=[];
for i=1:1000
M=5;
N=5;
S_2=randsample(S_X(2),N);
R=A(S_2,:);
X=A*pinv(R);
Er=[Er,norm(A-X*R,'fro')/nA];
end
semilogy((Er),'*')
%%%%%%%%%%%  CUR Approximation 
A=randn(500,5)*randn(5,500);
S_X=size(A);
nA=norm(A);
Er=[];
for i=1:1000
S_1=randsample(S_X(1),M);
S_2=randsample(S_X(2),N);
C=A(:,S_1);
R=A(S_2,:);
U=pinv(C)*A*pinv(R);
Er=[Er,norm(A-C*U*R,'fro')/nA];
end
semilogy((Er),'*')
%%%%%%%%%%%%CUR with intersection middle matrix
A=randn(500,5)*randn(5,500);
S_X=size(A);
nA=norm(A);
Er=[];
for i=1:1000
S_1=randsample(S_X(1),M);
S_2=randsample(S_X(2),N);
C=A(:,S_1);
R=A(S_2,:);
U=pinv(A(S_2,S_1));
Er=[Er,norm(A-C*U*R,'fro')/nA];
end
semilogy((Er),'*')

W=A(S_2,S_1);
norm(C*U*R-A,'fro')

norm(C*pinv(W)*R-A,'fro')

subplot(1,3,1)
imshow(uint8(C*U*R))
subplot(1,3,2)
imshow(uint8(C*pinv(W)*R))

[u,s,v]=svds(A,20);
subplot(1,3,3)
imshow(uint8(u*s*v'))
