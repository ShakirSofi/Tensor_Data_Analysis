function [P,Uinit,output,fit,obj_val] = cp_als_hybrid(X,R,varargin)
%CP_ALS Compute a CP decomposition of any type of tensor.
%
%   P = CP_ALS_HYBRID(X,R) computes an estimate of the best rank-R
%   CP model of a tensor X using the Hybrid-ALS algorithm, which is a
%   generalization of Orth-ALS.
%   The input X can be a tensor, sptensor, ktensor, or
%   ttensor. The result P is a ktensor.
%
%   Hybrid-ALS is a variant of standard ALS where the factor estimates are
%   orthogonalized at every iteration, upto a fixed number of iterations.
%   Subsequently, standard ALS updates are run. This is incontrast to Orth-ALS,
%   where orthogonalization is continued till the end. 
%
%   Reference: V. Sharan, G. Valiant, Orthogonalized ALS: A Theoretically 
%   Principled Tensor Decomposition Algorithm for Practical Use,
%   arXiv:1703.01804, 2017
%
%   P = CP_ALS_HYBRID(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'stop_orth' - Number of steps after which orthogonalization is to be
%      stopped {5}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%
%   The parameter stop_orth denotes the number of iterations after which to
%   stop orthogonalization. For example, to stop after 5 iterations, run
%   X_cpd = cp_als_hybrid(X_tensor, 30, 'stop_orth', 5);
%
%   [P,U0] = CP_ALS_HYBRID(...) also returns the initial guess.
%
%   [P,U0,out] = CP_ALS_HYBRID(...) also returns additional output that contains
%   the input parameters.
%
%   Note: The "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
%   loosely the proportion of the data described by the CP model, i.e., a
%   fit of 1 is perfect.
%
%   NOTE: Code modified from the original code for CP_ALS in the MATLAB Tensor Toolbox .
%   Updated in various minor ways per work of Phan Anh Huy. See Anh
%   Huy Phan, Petr Tichavský, Andrzej Cichocki, On Fast Computation of
%   Gradients for CANDECOMP/PARAFAC Algorithms, arXiv:1204.1586, 2012.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   P = cp_als_hybrid(X,2);
%   P = cp_als_hybrid(X,2,'stop_orth',5);
%   P = cp_als_hybrid(X,2,'dimorder',[3 2 1]);
%   P = cp_als_hybrid(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),[]}; %<-- Initial guess for factors of P
%   [P,U0,out] = cp_als_hybrid(X,2,'dimorder',[3 2 1],'init',U0);
%   P = cp_als(X,2,out.params); %<-- Same params as previous run
%
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

%% Extract number of dimensions and norm of X.
N = ndims(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParamValue('tol',1e-6,@isscalar);
params.addParamValue('stop_orth',5,@isscalar);
params.addParamValue('maxiters',30,@(x) isscalar(x) & x > 0);
params.addParamValue('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
stop_orth = params.Results.stop_orth;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
obj_val = zeros(maxiters,1);

%% Error checking

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end);
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    if strcmp(init,'random')
        %         fprintf('random\n');
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            %             Uinit{n} = rand(size(X,n),R);
            Uinit{n} = 2*rand(size(X,n),R)-1;
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs')
        %         fprintf('eigs\n');
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;

% P = ktensor([1]',U);
% save('tensor0', 'P');

if printitn>0
    fprintf('\nCP_ALS:\n');
end

%% Main Loop: Iterate until convergence

if (isa(X,'sptensor') || isa(X,'tensor')) && (exist('cpals_core','file') == 3)
    
    fprintf('Using C++ code\n');
    [lambda,U] = cpals_core(X, Uinit, fitchangetol, maxiters, dimorder);
    P = ktensor(lambda,U);
    
else
    
    UtU = zeros(R,R,N);
    for n = 1:N
        if ~isempty(U{n})
            UtU(:,:,n) = U{n}'*U{n};
        end
    end
    
    for iter = 1:maxiters
        
        fitold = fit;
        
        if iter <= stop_orth
            
            orth_const = 1;
            
            Q = U{1};
            t = size(Q);
            J = t(2);
            dim_max = t(1) -1 ;
            for n = 1:N
                Q = U{1};
                t = size(Q);
                if t(1)-1 <dim_max
                    dim_max = t(1) - 1;
                end
            end
            
            for n = 1:N
                Q = U{n};
                for i=1:J
                    Q(:,i) = Q(:,i)/norm(Q(:,i));
                    if i <= dim_max
                        for j=i+1:J
                            Q(:,j) = Q(:,j) - orth_const * Q(:,j)'*Q(:,i)*Q(:,i);
                        end
                    end
                end
                U{n} = Q;
            end
            
            for n = 1:N
                UtU(:,:,n) = U{n}'*U{n};
            end
            if printitn > 0
                fprintf('Orthogonalized,');
            end
            
        else
            if printitn > 0
                fprintf('Not Orthogonalized,');
            end
        end
        
        % Iterate over all N modes of the tensor
        for n = dimorder(1:end)
            
            % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = mttkrp(X,U,n);
            
            % Compute the matrix of coefficients for linear system
            Y = prod(UtU(:,:,[1:n-1 n+1:N]),3);
            Unew = Unew / Y;
            if issparse(Unew)
                Unew = full(Unew);   % for the case R=1
            end
            
            % Normalize each vector to prevent singularities in coefmatrix
            if iter == 1
                lambda = sqrt(sum(Unew.^2,1))'; %2-norm
            else
                lambda = max( max(abs(Unew),[],1), 1 )'; %max-norm
            end
            
            Unew = bsxfun(@rdivide, Unew, lambda');
            
            U{n} = Unew;
            UtU(:,:,n) = U{n}'*U{n};
            
        end
        
        P = ktensor(lambda,U);
        P = arrange(P);
        
        if normX == 0
            fit = norm(P)^2 - 2 * innerprod(X,P);
        else
            normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
            fit = 1 - (normresidual / normX); %fraction explained by model
        end
        fitchange = (fitold - fit);
        obj_val(iter) = fit;
        
        
        % Check for convergence
        if (iter > 1) && (abs(fitchange) < fitchangetol)
            flag = 0;
        else
            flag = 1;
        end
        
        if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
            fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, -fitchange);
        end
        
        % Check for convergence
        if (flag == 0)
            break;
        end
    end
end


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
P = fixsigns(P);

if printitn>0
    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(X,P);
    else
        normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
        fit = 1 - (normresidual / normX); %fraction explained by model
    end
    fprintf(' Final f = %e \n', fit);
end

obj_val = obj_val(1:iter);
output = struct;
output.params = params.Results;
output.iters = iter;
