% Run GAP and save parameters and solutions as reference test data
% to check if other algorithms are correct

numA = 10;
numY = 100;

sizesA{1} = [50 100];
sizesA{2} = [20 25];
sizesA{3} = [10 120];
sizesA{4} = [15 100];
sizesA{5} = [70 100];
sizesA{6} = [80 100];
sizesA{7} = [90 100];
sizesA{8} = [99 100];
sizesA{9} = [100 100];
sizesA{10} = [250 400];
for i = 1:numA sizesA{i} = fliplr(sizesA{i}); end

sigmamin = [0.00001 0.01 0.2 0.3 0.4 0.0001 0.1 0.001 0.1 0.1];

for i = 1:numA
    sz = sizesA{i};
    cellA{i} = randn(sz);
    m = round((0.2 + 0.6*rand)*sz(2));
    cellM{i} = randn(m,sz(2));
    cellY{i} = randn(m, numY);
    cellXinit{i} = zeros(sz(2), numY);
    for j = 1:numY
        cellEps{i}(j) = rand / 100; % restrict from 0 to 1% of measurements
    end
end
opt_num_iteration = 1000;
opt_greedy_level = 0.9;
opt_stopping_coefficient_size = 1e-4;
opt_l2solver = 'pseudoinverse';

%load GAPtestdata
tic
for i = 1:numA
    for j = 1:numY
        gapparams.num_iteration = opt_num_iteration;
        gapparams.greedy_level = opt_greedy_level;
        gapparams.stopping_coefficient_size = opt_stopping_coefficient_size;
        gapparams.l2solver = opt_l2solver;
        gapparams.noise_level = cellEps{i}(j);
        
        cellXr{i}(:,j) = GAP(cellY{i}(:,j), cellM{i}, cellM{i}', cellA{i}, cellA{i}', gapparams, cellXinit{i}(:,j));
    end
end
toc

save GAPtestdata