function Accu = R2LML(path,parameters)
%Implementation of R2LML
%Input:
%       path       -- loading path of your dataset
%       parameters -- structure containing all the hyperparamters
%Output:
%       Accu       -- K-nearest Neighbor accurancy result

%Load the dataset
load(path);
Dataset = train_data;

%Load the parameters
NumMa_K = parameters.NumMa_K; 
Lambda = parameters.lambda;
t0 = parameters.t0;
iter = parameters.iter;
epoch = parameters.epoch;
Kneigh = parameters.kneigh;

%Preprocessing
disp(['Propecessing......']);
[ToyM,ToyN] = size(Dataset);
[M,Ntr] = size(train_data); %Number of training patterns
[M,Ncr] = size(cross_data);
[M,Nte] = size(test_data);
D = M-1;%Dimension of the traing points

Label = Dataset(ToyM,:);
X = Dataset(1:D,:);

%Construct the similarity matrix S
S = zeros(Ntr,Ntr);
for i = 1:Ntr
    for j = (i+1):Ntr
        if Label(1,i) == Label(1,j)
            S(i,j) = 1;
        else
            S(i,j) = 0;
        end
    end
end
S = S+S'+eye(Ntr);

%Now it's time to minimize the problem
%Define the initial metric for each metric and the metric vector
g = zeros(NumMa_K,Ntr);
for k = 1:NumMa_K
    eval(['L',num2str(k),' = rand(D,D);']);
    if k < NumMa_K
        g(k,:) = (1/(NumMa_K-1))*rand(1,Ntr);
    end
end
g(NumMa_K,:) = ones(1,Ntr)-sum(g(1:(NumMa_K-1),:),1);
Index = 1;
while(1)
    fprintf('Epoch %3d: \n',Index);
    for k = 1:NumMa_K
        fprintf('Step 1. Proximal method for Metric L%d.\n',k);
        eval(['L',num2str(k),' = StepOne(L',num2str(k),',g(',num2str(k),',:),X,S,Lambda,t0,iter);']);
    end
    L_Con = [];
    for k = 1:NumMa_K
        eval(['L_Con = [L_Con,L',num2str(k),'];']);
    end
    g = StepTwo(X,S,NumMa_K,L_Con,g);
    Index = Index+1;
    if Index >= epoch
        break;
    end
    clc;
end
%Now we use the test data to test the best accurancy we have
clc;
L_Con = [];
for k = 1:NumMa_K
    eval(['L_Con = [L_Con,L',num2str(k),'];']);
end
disp(['***Testing Phase***'])
[label,accurancy] = Testknn(train_data,test_data,L_Con,NumMa_K,g,Kneigh);
% save label label;
disp(['***The final result of classification is ',num2str(accurancy*100),';***']);
Accu = accurancy;
