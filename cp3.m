function [A,B,C]= cp3(X,R,maxIter)
if nargin<3
    maxIter=500;
end


X=tensor(X);
sz1 = size(X);
A = rand(sz1(1), R);
B = rand(sz1(2), R);
C = rand(sz1(3), R);

snorm = norm(X);

for i =1:maxIter
  % For A
   M1 = ((C'*C).*(B'*B));
   khatA = khatrirao(C,B);
   A= double(tenmat(X,1))*khatA*pinv(M1);
   
   % For B
   M2 = ((C'*C).*(A'*A));
   khatB = khatrirao(C,A);
   B= double(tenmat(X,2))*khatB*pinv(M2);
   
   
   % For C
   M3 = ((B'*B).*(A'*A));
   khatC = khatrirao(B,A);
   C= double(tenmat(X,3))*khatC*pinv(M3);
  
   
   if mod(i,20)==0
       kt = ktensor({A,B,C});
       er = norm(tensor(X)-tensor(kt));
       fprintf('\n Iter: %d  with Fit %f', i, 1-er/snorm);
       
   end
end