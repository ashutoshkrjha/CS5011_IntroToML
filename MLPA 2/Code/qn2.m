Xtrain = mmread('Train_features');
Xtest = mmread('Test_features');
Ytest = mmread('Test_labels');
Ytrain = mmread('Train_labels');

m = size(Xtrain, 1);
M = 50; %Number of hidden layer neurons
K=4; %Number of output neurons

eta = 0.1; %Learning Rate
gamma = 1; %Regularization Parameter
Theta1 = rand(size(Xtrain,2)+1,M);
Theta2 = rand(M+1,K);
numiter = 0;
flag = 1;

gradwrtbeta = zeros(size(Theta1));
gradwrtalpha = zeros(size(Theta2));
sumtheta2 = zeros(size(Theta2_grad));

%Standardizing X

Xtrainmean = mean(Xtrain);
Xtestmean = mean(Xtest);
Xtrainstd = std(Xtrain);
Xteststd = std(Xtest);

Xtrainmean = repmat(Xtrainmean,m,1);
Xtestmean = repmat(Xtestmean,size(Xtest,1),1);
Xtrainstd = repmat(Xtrainstd,m,1);
Xteststd = repmat(Xteststd,size(Xtest,1),1);

Xtrain = (Xtrain-Xtrainmean)./(Xtrainstd);
Xtest = (Xtest-Xtestmean)./(Xteststd);

% Making the Y matrix
Ytest = repmat(Ytest,1,4);
Ytest((find(Ytest(:,1)==1)),:)=repmat([1 0 0 0],size(Ytest((find(Ytest(:,1)==1)),:),1),1);
Ytest((find(Ytest(:,1)==2)),:)=repmat([0 1 0 0],size(Ytest((find(Ytest(:,1)==2)),:),1),1);
Ytest((find(Ytest(:,1)==3)),:)=repmat([0 0 1 0],size(Ytest((find(Ytest(:,1)==3)),:),1),1);
Ytest((find(Ytest(:,1)==4)),:)=repmat([0 0 0 1],size(Ytest((find(Ytest(:,1)==4)),:),1),1);

Ytrain = repmat(Ytrain,1,4);
Ytrain((find(Ytrain(:,1)==1)),:)=repmat([1 0 0 0],size(Ytrain((find(Ytrain(:,1)==1)),:),1),1);
Ytrain((find(Ytrain(:,1)==2)),:)=repmat([0 1 0 0],size(Ytrain((find(Ytrain(:,1)==2)),:),1),1);
Ytrain((find(Ytrain(:,1)==3)),:)=repmat([0 0 1 0],size(Ytrain((find(Ytrain(:,1)==3)),:),1),1);
Ytrain((find(Ytrain(:,1)==4)),:)=repmat([0 0 0 1],size(Ytrain((find(Ytrain(:,1)==4)),:),1),1);


%Backprop algorithm
% Here a3 is output of final layer a2 is output of hidden layer and z's are
% the input to the layers

while (flag==1 | norm(gradwrtalpha)+norm(gradwrtbeta)>0.1 & numiter<500)
    flag=2;
    a1 = [ones(size(Xtrain,1),1) Xtrain];
    z2 = a1*Theta1;
    a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
    z3 = a2*Theta2;
    a3 = sigmoid(z3);
    h = a3;
%   delta = -(Ytrain-h).*h.*(ones(size(h))-h);
    delta = h-Ytrain; %Used cross entropy as it was giving better results
    gradwrtbeta = a2' * delta;
    S = a2.*(ones(size(a2))-a2).*(delta*Theta2');
    gradwrtalpha = a1' * S;
    gradwrtalpha = gradwrtalpha(:,2:M+1);
    Theta1 = Theta1 - eta*gradwrtalpha; %- 2*gamma*eta*[zeros(1,size(Theta1,2));Theta1(2:end,:)];
    Theta2 = Theta2 - eta*gradwrtbeta; %- 2*gamma*eta*[zeros(1,size(Theta2,2));Theta2(2:end,:)];
    numiter = numiter+1;
end

%Testing

a1 = [ones(size(Xtest,1),1) Xtest];
z2 = a1*Theta1;
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2*Theta2;
a3 = sigmoid(z3);
h = a3;
Yhat = double(bsxfun(@eq, h, max(h, [], 2)));

Ytest1 = Ytest(:,1);
Ytest2 = Ytest(:,2);
Ytest3 = Ytest(:,3);
Ytest4 = Ytest(:,4);

Yhat1 = Yhat(:,1);
Yhat2 = Yhat(:,2);
Yhat3 = Yhat(:,3);
Yhat4 = Yhat(:,4);

tp1 = sum(Yhat1==1 & Ytest1 == 1);
tn1 = sum(Yhat1==0 & Ytest1 == 0);
fp1 = sum(Yhat1==1 & Ytest1 == 0);
fn1 = sum(Yhat1==0 & Ytest1 == 1);
precision1 = tp1/(tp1+fp1)
recall1 = tp1/(tp1+fn1)
accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
F11 = 2*precision1*recall1/(precision1+recall1)

tp2 = sum(Yhat2==1 & Ytest2 == 1);
tn2 = sum(Yhat2==0 & Ytest2 == 0);
fp2 = sum(Yhat2==1 & Ytest2 == 0);
fn2 = sum(Yhat2==0 & Ytest2 == 1);
precision2 = tp2/(tp2+fp2)
recall2 = tp2/(tp2+fn2)
accuracy2 = (tp2+tn2)/(tp2+tn2+fp2+fn2)
F12 = 2*precision2*recall2/(precision2+recall2)

tp3 = sum(Yhat3==1 & Ytest3 == 1);
tn3 = sum(Yhat3==0 & Ytest3 == 0);
fp3 = sum(Yhat3==1 & Ytest3 == 0);
fn3 = sum(Yhat3==0 & Ytest3 == 1);
precision3 = tp3/(tp3+fp3)
recall3 = tp3/(tp3+fn3)
accuracy3 = (tp3+tn3)/(tp3+tn3+fp3+fn3)
F13 = 2*precision3*recall3/(precision3+recall3)

tp4 = sum(Yhat4==1 & Ytest4 == 1);
tn4 = sum(Yhat4==0 & Ytest4 == 0);
fp4 = sum(Yhat4==1 & Ytest4 == 0);
fn4 = sum(Yhat4==0 & Ytest4 == 1);
precision4 = tp4/(tp4+fp4)
recall4 = tp4/(tp4+fn4)
accuracy4 = (tp4+tn4)/(tp4+tn4+fp4+fn4)
F14 = 2*precision4*recall4/(precision4+recall4)