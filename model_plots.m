%% Train Test Split (80 / 20)
N = length(y);
idx = randperm(N);

Ntrain = round(0.8*N);

trainIdx = idx(1:Ntrain);
testIdx  = idx(Ntrain+1:end);

x1_tr = x1(trainIdx);  x1_te = x1(testIdx);
x2_tr = x2(trainIdx);  x2_te = x2(testIdx);
x3_tr = x3(trainIdx);  x3_te = x3(testIdx);

y_tr = y(trainIdx);
y_te = y(testIdx);

%% MODEL 1: Multiple Linear Regression
Xtr = [ones(size(x1_tr)) x1_tr x2_tr x3_tr];
Xte = [ones(size(x1_te)) x1_te x2_te x3_te];

b = regress(y_tr,Xtr);

y_pred_lin = Xte*b;

% Training Metrics (Linear Regression)
y_pred_lin_tr = Xtr*b;

RMSE_lin_tr = sqrt(mean((y_tr - y_pred_lin_tr).^2));
MAE_lin_tr  = mean(abs(y_tr - y_pred_lin_tr));
R2_lin_tr   = 1 - sum((y_tr - y_pred_lin_tr).^2)/sum((y_tr-mean(y_tr)).^2);

%% Metrics (Linear Regression)
RMSE_lin = sqrt(mean((y_te - y_pred_lin).^2));
MAE_lin  = mean(abs(y_te - y_pred_lin));
R2_lin   = 1 - sum((y_te - y_pred_lin).^2)/sum((y_te-mean(y_te)).^2);

%% MODEL 2: Decision Tree
I_tr = [x1_tr x2_tr x3_tr];
I_te = [x1_te x2_te x3_te];

Mdl = fitrtree(I_tr,y_tr);
y_pred_tree = predict(Mdl,I_te);

% Training Metrics (Decision Tree)
y_pred_tree_tr = predict(Mdl,I_tr);

RMSE_tree_tr = sqrt(mean((y_tr - y_pred_tree_tr).^2));
MAE_tree_tr  = mean(abs(y_tr - y_pred_tree_tr));
R2_tree_tr   = 1 - sum((y_tr - y_pred_tree_tr).^2)/sum((y_tr-mean(y_tr)).^2);

%% Metrics (Decision Tree)
RMSE_tree = sqrt(mean((y_te - y_pred_tree).^2));
MAE_tree  = mean(abs(y_te - y_pred_tree));
R2_tree   = 1 - sum((y_te - y_pred_tree).^2)/sum((y_te-mean(y_te)).^2);

%% Display Results

fprintf('\n==============================\n');
fprintf('MODEL PERFORMANCE\n');
fprintf('==============================\n');

%% Linear Regression
fprintf('\nMultiple Linear Regression:\n');

fprintf('--- Training Set ---\n');
fprintf('R2   = %.4f\n',R2_lin_tr);
fprintf('RMSE = %.4f\n',RMSE_lin_tr);
fprintf('MAE  = %.4f\n',MAE_lin_tr);

fprintf('--- Testing Set ---\n');
fprintf('R2   = %.4f\n',R2_lin);
fprintf('RMSE = %.4f\n',RMSE_lin);
fprintf('MAE  = %.4f\n',MAE_lin);

%% Decision Tree
fprintf('\nDecision Tree:\n');

fprintf('--- Training Set ---\n');
fprintf('R2   = %.4f\n',R2_tree_tr);
fprintf('RMSE = %.4f\n',RMSE_tree_tr);
fprintf('MAE  = %.4f\n',MAE_tree_tr);

fprintf('--- Testing Set ---\n');
fprintf('R2   = %.4f\n',R2_tree);
fprintf('RMSE = %.4f\n',RMSE_tree);
fprintf('MAE  = %.4f\n',MAE_tree);

%%  Regression & Prediction Plots
% -------- Multiple Regression --------

% Predictions on training data
y_pred_lin_tr = Xtr*b;

figure
subplot(1,3,1)
scatter(y_tr,y_pred_lin_tr,'filled')
lsline
xlabel('Actual Tavg')
ylabel('Predicted Tavg')
title('Multiple Regression (Training)')
grid on

subplot(1,3,2)
scatter(y_te,y_pred_lin,'filled')
lsline
xlabel('Actual Tavg')
ylabel('Predicted Tavg')
title('Multiple Regression (Testing)')
grid on

subplot(1,3,3)
plot(y_te,'b'); hold on
plot(y_pred_lin,'r')
legend('Actual','Predicted')
xlabel('Observation number')
ylabel('Tavg')
title('Multiple Regression Prediction')
grid on

%% -------- Decision Tree --------

% Predictions on training data
y_pred_tree_tr = predict(Mdl,I_tr);

figure

subplot(1,3,1)
scatter(y_tr,y_pred_tree_tr,'filled')
lsline
xlabel('Actual Tavg')
ylabel('Predicted Tavg')
title('Decision Tree (Training)')
grid on

subplot(1,3,2)
scatter(y_te,y_pred_tree,'filled')
lsline
xlabel('Actual Tavg')
ylabel('Predicted Tavg')
title('Decision Tree (Testing)')
grid on

subplot(1,3,3)
plot(y_te,'b'); hold on
plot(y_pred_tree,'r')
legend('Actual','Predicted')
xlabel('Observation number')
ylabel('Tavg')
title('Decision Tree Prediction')
grid on

