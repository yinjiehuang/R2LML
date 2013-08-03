function Output = StepOne(L,g,X,S,lambda,t0,iter)
%Step One of R2LML, train different metric
%Meanwhile, the group vector g is fixed
%Input:
%       L       -- Distance metric for group 
%       g       -- Group vector (fixed here)
%       X       -- Training Data
%       S       -- Similarity Matrix
%       lambda  -- Regularization paramter of nuclear norm
%       t0      -- Step length 
%       iter    -- Number of iterations 
%Output:
%       Output  -- metric learned

%Initilize some parameters
epsilon = 1e-5;
Beta = 0.5; %Line search step

[XM,XN] = size(X);
D = XM;
N = XN;
[LM,LN] = size(L);

Lyk = L;
Lxk = L;
Index = 1; %This index is used in controlling number of minimization steps
G = repmat(g,N,1).*repmat(g',1,N);
%Now we are trying to minimize the problem
%The problem formation is like f(x)+h(x), in which f(x) is
%differentiable and h(x) is non-differentiable
while(1)
    %First of all, let's compute gradient of f(x)
   %Use bfxfun to accelerate the computation of the metric
    XX = Lyk*X;
    E = bsxfun(@plus, sum(XX.*XX,1)',(-2)*XX'*XX);
    E = bsxfun(@plus, sum(XX.*XX,1),E);
    fk = sum(sum(S.*E.*G+(1-S).*max(0,1-E)));
    
    X_Temp = repmat(X,1,N);
    Coff = S.*G-(1-S).*sign(max(0,1-E));
    temp_S = repmat(reshape(Coff',1,N*N),D,1);
    Gra_f = 2*(temp_S.*X_Temp)*X_Temp'-2*(temp_S.*reshape(repmat(X,N,1),D,N*N))*repmat(X',N,1);
    clear X_Temp Coff temp_S;
    Gra_f = 2*Lyk*Gra_f;
    
    tk = t0;
    
    Lxk_update = Prox_Nuclear(Lyk-tk*Gra_f,lambda*tk);
    Lyk = Lxk_update+(Index-1)/(Index+2)*(Lxk_update-Lxk);
    Lxk = Lxk_update;
    %Check Convergence
    if mod(Index,10) == 0
        fprintf('Iters %5d. Max Gradient %15f. Cost function %15f.\t \n',Index,abs(max(max(Gra_f))),fk);
    end
    if abs(max(max(Gra_f))) <= epsilon || Index >= iter
        break;
    end
    Index = Index+1;
end
Output = Lyk;