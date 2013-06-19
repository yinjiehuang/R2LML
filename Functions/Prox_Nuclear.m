function Output = Prox_Nuclear(X,t)
%Compute the proximal projection of Nuclear norm

[P,Sigma,Q] = svd(X);
[M,N] = size(Sigma);
n = min(M,N);

New_Sigma = zeros(M,N);
for i = 1:n
    if Sigma(i,i) > t
        New_Sigma(i,i) = Sigma(i,i)-t;
    elseif Sigma(i,i) < -t
        New_Sigma(i,i) = Sigma(i,i)+t;
    else
        New_Sigma(i,i) = 0;
    end
end

Output = P*New_Sigma*Q';