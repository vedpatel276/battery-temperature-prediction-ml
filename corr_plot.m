clc; clear all; close all;
T = readtable("BC_35.csv");

% Dropping the variables with zero variance
T.BC=[];
T.Chg_DchgCycle = [];
V1 = T.Properties.VariableNames;

% Plotting the heatmap
R1 = corr(T{:,:}, 'Type', 'Pearson');
figure;
H1 = heatmap(V1,V1,R1); 
title("Pearson Correlation Matrix Heatmap");

R2 = corr(T{:,:},'Type','Spearman');
figure;
H2 = heatmap(V1, V1, R2);
title("Spearman Correlation Matrix Heatmap");


