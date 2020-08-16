%task 7

%creates the linear regresion model from acceleration vs mpg
task7_mdl = fitlm((data_train2(:,6)), (data_train2(:,1)));

%gets the intercept and X1 values from linear regresion model
task7_intercept = task7_mdl.Coefficients{1,1};
task7_x1 = task7_mdl.Coefficients{2,1};

%gets the predicted mpg from linear regresion model
task7_predicted = (data_train2(:,6) * task7_x1) + task7_intercept;

%gets the MSE from the predicted mpg train data
task7_mse = sum(((task7_predicted(:,1) - data_train2(:,1)).^2)/size(task7_predicted,1));


%task 8

%using the intercept and X1 from the model to get a calculated list for the
%data test mpg
task8_predicted = (data_test2(:,6) * task7_x1) + task7_intercept;

%gets the mse from the mpg test data
task8_mse = sum(((task8_predicted(:,1) - data_test2(:,1)).^2)/size(task8_predicted,1));

%creates a scatter plot between true and predicted mpg
scatter(data_test2(:,1),task8_predicted(:,1)); 


%task 9

%creates the linear regresion model from horsepower vs mpg
task9_mdl = fitlm((data_train2(:,4)), (data_train2(:,1)));

%gets the intercept and X1 values from linear regresion model
task9_intercept = task9_mdl.Coefficients{1,1};
task9_x1 = task9_mdl.Coefficients{2,1};

%gets the predicted mpg from linear regresion model
task9_predicted = (data_train2(:,4) * task9_x1) + task9_intercept;

%gets the MSE from the predicted mpg train data
task9_mse = sum(((task9_predicted(:,1) - data_train2(:,1)).^2)/size(task9_predicted,1));


%task 10

%using the intercept and X1 from the model to get a calculated list for the
%data test mpg
task10_predicted = (data_test2(:,4) * task9_x1) + task9_intercept;

%gets the mse from the mpg test data
task10_mse = sum(((task10_predicted(:,1) - data_test2(:,1)).^2)/size(task10_predicted,1))

%creates a scatter plot between true and predicted mpg
scatter(data_test2(:,1),task10_predicted(:,1));


%task 11

%creates the linear regresion model from weight vs horsepower
task11_mdl = fitlm((data_train2(:,5)), (data_train2(:,4)));

%gets the intercept and X1 values from linear regresion model
task11_intercept = task11_mdl.Coefficients{1,1};
task11_x1 = task11_mdl.Coefficients{2,1};

%gets the predicted weight from linear regresion model
task11_predicted = (data_train2(:,5) * task11_x1) + task11_intercept;

%gets the MSE from the predicted weight train data
task11_mse = sum(((task11_predicted(:,1) - data_train2(:,4)).^2)/size(task11_predicted,1));


%task 12

%using the intercept and X1 from the model to get a calculated list for the
%data test weight
task12_predicted = (data_test2(:,4) * task11_x1) + task11_intercept;

%gets the mse from the weight test data
task12_mse = sum(((task12_predicted(:,1) - data_test2(:,1)).^2)/size(task12_predicted,1))

%creates a scatter plot between true and predicted horsepower
scatter(data_test2(:,4),task12_predicted(:,1));

