function Tpsi = gen_caf(u,X,d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate CAF tensor as order-d derivative of the second generating
% charateristic function of X at the vector u
% Tpsi is an order d tensor of size n x n x ... x n 
% where n is number of rows of X
% each row X(k,:) is a signal
%
% Phan Anh Huy, 2016
y = X;
[n,T] = size(y);
w = exp(u'*y);
phi = mean(w); % phix_(u) 
dphi = cell(d,1);
dphi{1} = y*w'/T; % k-th order derivatives of Phi_x(u) 
for k = 2:d
    dphi{k} = reshape(khatrirao(repmat({y},1,k))*w'/T,n*ones(1,k));
end

pmax = d;
Tpsi = dphi{d}/phi;

for kp = 2:pmax %  kp: number of entries in a partition of 1:d
    part_k = partitions(d,kp);
    no_entries  = cell2mat(cellfun(@(x) cellfun(@numel,x),part_k,'uni',0));
    no_entries = sortrows(sort(no_entries,2));
    [no_entries_,ii,jj] = unique(no_entries,'rows');
    inx = find(diff([jj;jj(end)+1]));
    no_parts_kp = diff([0;inx]); % number of partitions which have the same number of entries
    % no_parts_kp(i) is the number of partitions as in No.entries(i,:)
    
    
    dphi_kp = 0;
    for k_gr = 1:size(no_entries_,1)
        ii = no_entries_(k_gr,:);
        % construct dphi_i1i2...i(kp) 
        dphi_ii = full(ktensor(cellfun(@(x) x(:),dphi(ii),'uni',0)));
        dphi_ii = reshape(dphi_ii,n*ones(1,d));
        % symmetrisize;
        %dphi_ii = double(symmetrize(tensor(dphi_ii)));
        dphi_ii = double(symmetrize(double(dphi_ii)));
        dphi_kp = dphi_kp + dphi_ii * no_parts_kp(k_gr);
    end
    
    %% Add into the total CAF tensor 
    Tpsi = Tpsi + (-1)^(kp-1) * factorial(kp-1) * dphi_kp/phi^kp;
end
end


%% 
function Ts = symmetrize(T)
% T : tensor of size I x I x... x I
persistent Pmat order I
I_t = size(T,1);
order_t = ndims(T);
if isempty(Pmat) || (order~= order_t) || ( I ~= I_t)
    order = order_t;
    I = I_t;
    Pmat = gen_symmetrize_matrix(I,order);
end

Ts = reshape(Pmat*T(:),size(T));
end

%% generate permutation matrix for tensor symmetricization 
function Pmat = gen_symmetrize_matrix(I,order)
% tensor of size I x I x ... x I and order 
sym_perms = perms(1:order);
ix = (sum(abs(bsxfun(@minus,sym_perms,1:order)),2)==0);
sym_perms(ix,:) = [];
total_perms = size(sym_perms,1);

K = I^order;
P = speye(K,K);
% Create an average tensor
Pi = reshape(1:K,I*ones(1,order));
Pmat = P;
for i = 1:total_perms
    Pi_x = permute(Pi,sym_perms(i,:));
    Pmat = Pmat + P(:,Pi_x(:));
end
Pmat = Pmat/(total_perms+1);
end
%%

function C = partitions(M,K)
%PARTITIONS Find all partitions of a set.
% C = PARTITIONS(N), for scalar N, returns a cell array, wherein each cell 
% has a various number of other cells representing the partitions of the  
% set of numbers {1,2,3,...N}.  The length of C is the Bell number for N.
% C = PARTITIONS(N), for vector N, returns the partitions of the vector
% elements treated as members of a set.
% C = PARTITIONS(N), for cell N, returns the partitions of the cell
% elements treated as members of a set.
% PARTITIONS(N,K) returns only the partitions of which have K members.  
%
% K must be less than or equal to N for scalar N, or length(N).
%
%
% Examples:
%
%     C = partitions(4); % Find the partitions of set {1 2 3 4}
%     home  % Makes for nice display for small enough N.  Else use clc.
%     partdisp(C) % File accompanying PARTITIONS
%
%     C = partitions({'peanut','butter','yummy'});
%     home
%     partdisp(C)
%
%     C = partitions([5 7 8 50],3); % partitions of set {5 7 8 50}
%     home
%     partdisp(C,3)
%
%     C = partitions(['a','b','c','d']); % for longer chars, use {}, not []
%     home
%     partdisp(C)
%
% Class support for inputs N,K:
%      float: double, single, char (N only)
% 
% See also, nchoosek, perms, npermutek (On the FEX)
%
% Author: Matt Fig,   
% Contact: popkenai@yahoo.com
% Date: 5/17/2009

Kflg = false;  % No K passed in.
Cflg = false;  % A set not passed in.

if length(M)>1 
    N = length(M); % User passed {'here','there','everywhere'} for example.
    Cflg = true;
else
    if iscell(M)  % User passed {2} for example.
        error('Set arguments must have more than one element.')
    end
    
    N = M;
end

if nargin>1
    Kflg = true;  % Used in while loop below, K passed in.
    S = stirling(N,K);  % The number of partitions, Stirling number.
    C = cell(S,min(1,K)); % Main Cell.
    cnt = 1;  % Start the while loop counter.
else
    K = 0;  % Since user doesn't want only certain partitions.  
    S = ceil(sum((1:2*N).^N./cumprod(1:2*N))/exp(1)); % Bell number.
    C = cell(S,1); % Main Cell.
    
    if Cflg
        C{1} = {M};
    else
        C{1} = {1:N}; % First one is easy.
    end
    
    cnt = 2; % Start the while loop counter.
end

if K~=ceil(K) || numel(K)~=1  
    error('Second argument must be an integer of length 1.  See help.')
end

if N<0 || K>N || K<0
   error('Arguments must be greater than zero, and K cannot exceed N.') 
end

if N==0
    C = {{}}; % Easy case.
    return
end

if K==1
    if Cflg
        C{1} = {M};  % Easy case.
    else
        C{1} = {1:N};  % Easy case.
    end
    return
end

if Cflg
    NV = M;
else
    NV = 1:M; % Vector: base partition, RGF indexes into this guy.
end

NU = 1; % Number of unique indices in current partition.
stp1 = N; % Controls assigning of indices.
RGF = ones(1,N); % Holds the indexes.
BLD = cell(1,N); % Smaller cell array will be used in creating larger.

while cnt<=S
    idx1 = N; % Index into RGF.
    stp2 = RGF(idx1); % Works with stp1.

    while stp1(stp2)==1
        RGF(idx1) = 1;  % Assign value to RGF.
        idx1 = idx1 - 1; % Need to increment idx1 for translation below.
        stp2 = RGF(idx1); % And set this guy for stp1 assign below.
    end

    NU = NU + idx1 - N;  % Get provisional number of unique vals.
    stp1(1) = stp1(1) + N - idx1;

    if stp2==NU % Increment the number of unique elements.
        NU = NU +1;
        stp1(NU) = 0;
    end

    RGF(idx1) = stp2 + 1;  % Increment this position.
    stp1(stp2) = stp1(stp2) - 1;  % Translate indices of these two. 
    stp1(stp2+1) = stp1(stp2+1) + 1; % Re-assign stopper.

    if NU==(~Kflg * NU + Kflg * K)
    % We could use:  C{cnt} = accumarray(RGF',NV',[],@(x) {x});   (SLOW!!)
    % or the next lines to C{cnt} = TMP;
        TMP = BLD(1:NU); % Build subcell of correct size.
        TMP{1} = NV(RGF==1); % These first two are always here.... no loop.
        TMP{2} = NV(RGF==2);

        for ii = 3:NU % Build the rest of cell array, if any.
            TMP{ii} = NV(RGF==ii);
        end

        C{cnt} = TMP;  % Assign cell at jj. 
        cnt = cnt + 1;
    end
end

end

function S = stirling(N,K)
% Calculate the Stirling number of the second kind. Subfunc to partitions.

for ii = K:-1:0
    S(ii+1) = (-1)^ii * prod(1:K) / (prod(1:(K-ii)) *...
              (prod(1:ii)))*(K - ii)^N;  %#ok
end

S = 1/prod(1:K) * sum(S);
end