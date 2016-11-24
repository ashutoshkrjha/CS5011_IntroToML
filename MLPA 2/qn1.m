Xtrain = mmread('Train_features');
Xtest = mmread('Test_features');
Ytest = mmread('Test_labels');
Ytrain = mmread('Train_labels');

% Ytest = repmat(Ytest,1,4);
% Ytest((find(Ytest(:,1)==1)),:)=repmat([1 0 0 0],size(Ytest((find(Ytest(:,1)==1)),:),1),1);
% Ytest((find(Ytest(:,1)==2)),:)=repmat([0 1 0 0],size(Ytest((find(Ytest(:,1)==2)),:),1),1);
% Ytest((find(Ytest(:,1)==3)),:)=repmat([0 0 1 0],size(Ytest((find(Ytest(:,1)==3)),:),1),1);
% Ytest((find(Ytest(:,1)==4)),:)=repmat([0 0 0 1],size(Ytest((find(Ytest(:,1)==4)),:),1),1);
% 
% Ytrain = repmat(Ytrain,1,4);
% Ytrain((find(Ytrain(:,1)==1)),:)=repmat([1 0 0 0],size(Ytrain((find(Ytrain(:,1)==1)),:),1),1);
% Ytrain((find(Ytrain(:,1)==2)),:)=repmat([0 1 0 0],size(Ytrain((find(Ytrain(:,1)==2)),:),1),1);
% Ytrain((find(Ytrain(:,1)==3)),:)=repmat([0 0 1 0],size(Ytrain((find(Ytrain(:,1)==3)),:),1),1);
% Ytrain((find(Ytrain(:,1)==4)),:)=repmat([0 0 0 1],size(Ytrain((find(Ytrain(:,1)==4)),:),1),1);

model = svmtrain(Ytrain, Xtrain);
[predicted_label, accuracy, prob_estimates] = svmpredict(Ytest, Xtest, model);