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

%We also need to preprocessing the combined metric
for k=1:NumG
     eval(['L',num2str(k),'=L(:,(k-1)*D+1:D*k);']);
end    

%Find the nearest 1 neighbour in the training dataset of testing dataset,
%and set the group information vector on this testing point
testing_g = [];
for i = 1:Ntest
    tempdis = [];
    for j = 1:Ntrain
        tempdis(j) = (testing(:,i)-training(:,j))'*(testing(:,i)-training(:,j));
    end
    index = find(tempdis == min(tempdis));
    testing_g(:,i) = g(:,index(1));
end

for i = 1:Ntest
    tempdis = zeros(1,Ntrain);
    for j = 1:Ntrain
        %Now i is the point in the test data and j is in the training set,
        %let's see the distance between all the j and i
        tempmetric = zeros(D,D);
        for k = 1:NumG
            eval(['L',num2str(k),' = L(:,(k-1)*D+1:D*k);']);
            eval(['Ltemp = L',num2str(k),';']);
            tempmetric = tempmetric+Ltemp'*Ltemp*g(k,j)*testing_g(k,i);
        end
        tempdis(j) = (testing(:,i)-training(:,j))'*tempmetric*(testing(:,i)-training(:,j));
    end
    tempsort = sort(tempdis);
    IndexKnn = zeros(1,Kneigh);
    templabel = zeros(1,Kneigh);
    for p = 1:Kneigh
        hehe = find(tempdis == tempsort(p));
        IndexKnn(p) = hehe(1);
        templabel(p) = trainlabel(IndexKnn(p));
    end
    temptestlabel = maxelement(templabel);
    tempindex = randperm(length(temptestlabel));
    testresultlabel(i) = temptestlabel(tempindex(1));
end

%Let's find the classification errors based on the prediction
hehe = testlabel-testresultlabel;
result = 1-length(find(hehe~=0))/Ntest;
