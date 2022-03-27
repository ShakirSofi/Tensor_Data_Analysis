function [A,B,C,lam]=cp3d(X,R)
Iter1=500; Iter2=3;
A=randn(size(X,1),R);B=randn(size(X,2),R);C=randn(size(X,3),R);
for i=1:Iter1
    for r=1:R
        ind=setdiff(1:R,r);
        lam=lam_find(X,A(:,ind),B(:,ind),C(:,ind));
        EST=tensor_reconst(A(:,ind),B(:,ind),C(:,ind),lam);
        for i2=1:Iter2
        A(:,r)=tens2mat(X-EST,1)/(kr(B(:,r),C(:,r))');
        B(:,r)=tens2mat(X-EST,2)/(kr(A(:,r),C(:,r))');
        C(:,r)=tens2mat(X-EST,3)/(kr(A(:,r),B(:,r))');
        end 
    end
    if mod(i, 20)==0
        fprintf('\n Iter: %d', i)
    end
end
lam=lam_find(X,A,B,C);
%%%%%% FUNCTIONS %%%%%%%%%
function Xhat=tensor_reconst(A,B,C,lam)
Xhat=(A*diag(lam))*(kr(C,B)');
Xhat=reshape(Xhat,[size(A,1),size(B,1),size(C,1)]);
 %%%%%%%%%%%%%%%%%%%%%%
function lam=lam_find(X,A,B,C)
for r=1:size(A,2)
   D(:,r)=vec(tensor_reconst(A(:,r),B(:,r),C(:,r),1));
end
lam=pinv(D)*X(:);
 %%%%%%%%%%%%%%%%%%%%%%
function [X_mat]=tens2mat(X,mode)
ORDERS=[1 3 2;2 3 1;3 2 1];
X_mat=reshape(permute(X,ORDERS(mode,:)),size(X,mode),numel(X)/size(X,mode));
 %%%%%%%%%%%%%%%%%%%%%%
function v = vec(x)
v = reshape(x,numel(x),1);
 %%%%%%%%%%%%%%%%%%%%%%
function AB = kr(A,B)
[I,F]=size(A);[J,~]=size(B);AB=zeros(I*J,F);
for f=1:F
    AB(:,f)=vec(B(:,f)*A(:,f).');
end