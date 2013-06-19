function Output = StepTwo(X,S,K,L,input_g)
%Step two of R2LML, fixed metric, compute vector g
%Input:
%       X  -- Training Data
%       S  -- Similarity matrix
%       K  -- Number of Groups
%       L  -- L matrix concatenated together
%       input_g  -- current group matrix

[D,N] = size(X);

%First of all, we need to process the input information\
S_Til = zeros(N*K,N*K);
g = zeros(K*N,1);
for k = 1:K
    eval(['L',num2str(k),' = L(:,(k-1)*D+1:D*k);']);
    eval(['Ltemp = L',num2str(k),';']);
    Temp = zeros(N,N);
    for m = 1:N
        for n = (m+1):N
            Emn = (X(:,m)-X(:,n))'*Ltemp'*Ltemp*(X(:,m)-X(:,n));
            Temp(m,n) = S(m,n)*Emn;
        end
    end
    Temp = Temp+Temp';
    %Based on S_Ba, we need to get S_Til matrix
    S_Til((k-1)*N+1:k*N,(k-1)*N+1:k*N) = Temp;
    %Then let's vectorize the g
    g((k-1)*N+1:k*N,1) = input_g(k,:)';
end

%Let's construct the majorization function and do the binary search for g
mu = -max(eig(S_Til))-1;
H = S_Til+mu*eye(K*N);
B = kron(ones(1,k),eye(N));
gprime = g;
Index = 1;
while(1)
    Phi = (gprime'*H)';
    %Do the binary search to solve the majority function
    %We need to do binary search for every a entry
    for i = 1:N
        %Get the corresponding index
        Indexg = zeros(K,1);
        for k = 1:K
            Indexg(k) = (k-1)*N+i;
        end
        a1 = -max(abs(Phi))*10;
        a2 = max(abs(Phi))*10;
        while(1)
            a = (a1+a2)/2;
            temp = 0;
            for k = 1:K
                gk(Indexg(k),1) = -1/(2*mu)*max(0,a-Phi(Indexg(k)));
                temp = temp+gk(Indexg(k));
            end
            diff = temp-1;
            if abs(diff) <= 1e-3
                break;
            elseif diff >= 0
                a2 = a;
            else
                a1 = a;
            end
        end
    end
    %Now check the convergence
    qk_1 = gprime'*S_Til*gprime;
    qk = gk'*S_Til*gk;
    Difference = abs(qk-qk_1);
    if mod(Index,10) ==0
        fprintf('Iter %5d. Difference of Majority: %15f. \t \n',Index,Difference);
    end
    if Difference <= 1e-3 || Index >= 3000
        break;
    else
        gprime = gk;
        Index = Index+1;
    end
end

%Reformulate the output
Output = zeros(K,N);
for k = 1:K
    Output(k,:) = gk(((k-1)*N+1):(k*N),1);
end
