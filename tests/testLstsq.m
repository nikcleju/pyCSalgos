% test how much time solving a system takes in Matlab

clear all
close all

load testLstsq

tic
for i = 1:nruns
    A \ b;
end
toc