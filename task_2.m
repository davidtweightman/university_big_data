%read in the data set
data = xlsread('clinicalfeatures.xlsx');

%splits the data by creating a list of bools
datasplit = cvpartition(size(data,1),'HoldOut',0.3);
datasplitbool = datasplit.test;
% Separate to training and test data
datatrain = data(~datasplitbool,:);
datatest  = data(datasplitbool,:);

clear datasplitbool cv datasplit;

%splits the data into more usable matrixes
traintarget = datatrain(:,10);
datatrain2 = datatrain(:,1:9);

testtarget = datatest(:,10);
datatest2 = datatest(:,1:9);


%trains support vector model
SVMmodel = fitcsvm(datatrain2, traintarget);
%gets predictions from model
SVMpredict = predict(SVMmodel, datatest2);

%calculates number of correct predictions
correct = 0;
for i = 1 : size(testtarget, 1)
    if SVMpredict(i) == testtarget(i)
        correct = correct + 1;
    end
end

%calculates error rate of SVM model
SVMerror = (correct / size(testtarget, 1));

clear i correct


%trains decision tree model
DTmodel = fitctree(datatrain2, traintarget);
%gets predictions for model
DTpredict = predict(DTmodel, datatest2);

%calculates number of correct predictions
correct = 0;
for i = 1 : size(testtarget, 1)
    if DTpredict(i) == testtarget(i)
        correct = correct + 1;
    end
end

%calculates error rate of SVM model
DTerror = (correct / size(testtarget, 1));

clear i correct



