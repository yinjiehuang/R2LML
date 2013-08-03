function [testresultlabel,result] = Testknn(traindata,testdata,L,NumG,g,Kneigh)
%Testing using k-nearest neighbors
%Input:
%       traindata  -- Training data 
%       testdata   -- Testing data
%       L          -- Combined metric
%       NumG       -- Number of groups
%       g          -- Group vectors
%       Kneigh     -- Number of nearest neighbors want to use

%We need to split the label and data part of the test and training data
[TempD,Ntrain] = size(traindata);
[TempD,Ntest] = size(testdata);
D = TempD-1;
training = traindata(1:D,:);
trainlabel = traindata(TempD,:);
testing = testdata(1:D,:);
testlabel = testdata(TempD,:);

testresultlabel = zeros(1,Ntest);

%Find the nearest 1 neighbour in the training dataset of testing dataset,
%and set the group information vector on this testing point
M = bsxfun(@plus, sum(testing .* testing, 1)', (-2) * testing' * training);        
M = bsxfun(@plus, sum(training .* training, 1), M);
M(M<0)=0;
[Mdis,index] = min(M,[],2);
testing_g = g(:,index);
clear M index Mdis;

tempmetric = zeros(D*Ntest,D*Ntrain);
for k=1:NumG
    eval(['L',num2str(k),'=L(:,(k-1)*D+1:D*k);']);
    eval(['Ltemp=L',num2str(k),';']);
    LL = Ltemp'*Ltemp;
    tempg = repmat(testing_g(k,:),D,1);
    tempg2 = repmat(g(k,:),D,1);
    temp1 = bsxfun(@times,tempg(:),repmat(LL,Ntest,Ntrain));
    tempmetric = tempmetric+bsxfun(@times,tempg2(:)',temp1);
end
clear tempg tempg2 temp1;
for m=1:Ntest
    for n=1:Ntrain
        temp = tempmetric(D*(m-1)+1:D*m,D*(n-1)+1:D*n);
        dist(m,n) = (testing(:,m)-training(:,n))'*temp*(testing(:,m)-training(:,n));
    end
end
clear tempmetric;
 
[Y,I] = sort(dist,2);
clear Y;
testresultlabel = mode(trainlabel(I(:,1:Kneigh)),2);

%Let's find the classification errors based on the prediction
hehe = testlabel-testresultlabel';
result = 1-length(find(hehe~=0))/Ntest;