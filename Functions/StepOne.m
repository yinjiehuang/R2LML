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

%Now we are trying to minimize the problem
%The problem formation is like f(x)+h(x), in which f(x) is
%differentiable and h(x) is non-differentiable
while(1)
    %First of all, let's compute gradient of f(x)
    Gra_f = zeros(LM,LN);
    fk = 0;
    for m = 1:N
        for n = (m+1):N
            Emn = (X(:,m)-X(:,n))'*Lyk'*Lyk*(X(:,m)-X(:,n));
            fk = fk+S(m,n)*Emn*g(n)*g(m)+(1-S(m,n))*max(0,1-Emn);
            Cmn = (X(:,m)-X(:,n))*(X(:,m)-X(:,n))';
            if Emn <= 1
                h_pri = 1;
            else
                h_pri = 0;
            end
            Gra_f = Gra_f+2*Lyk*(S(m,n)*Cmn*g(n)*g(m)-(1-S(m,n))*Cmn*h_pri);
        end
    end
    fk = 2*fk; %Since the matrix is symmetric, we could use this to reduce the complexity
    Gra_f = 2*Gra_f;
    
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