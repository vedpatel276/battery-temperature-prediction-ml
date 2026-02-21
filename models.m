clc; clear; close all;
T = readtable("BC_35.csv");

% Inputs
x1 = T.DischargeCurrent;
x2 = T.Vact;
x3 = T.Ah;
% Output
y = T.Tavg;

%% Remove NaN rows (important)
M = [x1 x2 x3 y];
M = M(all(~isnan(M),2),:);

x1 = M(:,1);
x2 = M(:,2);
x3 = M(:,3);
y  = M(:,4);

%% Multiple Linear Regression
X = [ones(size(x1)) x1 x2 x3];
b = regress(y,X);

%% 3D Scatter (only x1,x2,y can be visualized)
figure
scatter3(x1,x2,y,'filled')
hold on

%% Create grid
x1fit = linspace(min(x1),max(x1),40);
x2fit = linspace(min(x2),max(x2),40);

[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);

% Fix Ah at rms value for visualization
x3rms= rms(x3);

%% Regression surface 
YFIT = b(1)+ b(2).*X1FIT+ b(3).*X2FIT+ b(4).*x3rms.*ones(size(X1FIT));

%% Plot surface
mesh(X1FIT,X2FIT,YFIT)

xlabel('Discharge Current')
ylabel('Vact')
zlabel('Tavg')

title('Multiple Linear Regression: Tavg Prediction')
view(50,10)
grid on
hold off

%%  Decision Tree
I = [x1 x2 x3];
Mdl = fitrtree(I,y);
view(Mdl,'Mode','graph')

