%This Demo shows how to run R2LML 
clear;clc;
%%Read the data
path = [pwd,'\Data\ionosphere'];
addpath([pwd,'\Functions']);

%%We will set all the hyperparamters here
%Number of metric
parameters.NumMa_K = 7;
%Regularization value lambda
parameters.lambda = 1;
%Step length of PSD
parameters.t0 = 1e-5;
%Number of steps of PSD for each metric
parameters.iter = 200;
%Number of epoches of two steps
parameters.epoch = 5;
%Number of k-nearest neighbors when testing
parameters.kneigh = 5;


%%Run the algorithm
accu = R2LML(path,parameters);
