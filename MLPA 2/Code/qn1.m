Xtrain = mmread('Train_features');
Xtest = mmread('Test_features');
Ytest = mmread('Test_labels');
Ytrain = mmread('Train_labels');

m = size(Xtrain, 1);
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

%Change the model parameters here to get the various answers
model1 = svmtrain(Ytrain, Xtrain,'-t 0 -c 0.1');
model2 = svmtrain(Ytrain, Xtrain,'-t 1 -c 3 -d 1');
model3 = svmtrain(Ytrain, Xtrain,'-t 2 -c 1 -g 0.01');
model4 = svmtrain(Ytrain, Xtrain,'-t 3 -c 0.1 -g 0.01 -r 0');

save('ME14B148','model1','model2','model3','model4');
% [predicted_label, accuracy, prob_estimates] = svmpredict(Ytest, Xtest, model);