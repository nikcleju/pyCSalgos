% Run SL0 and save parameters and solutions as reference test data
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

sigmamin = [0.00001 0.01 0.2 0.3 0.4 0.0001 0.1 0.001 0.1 0.1];

for i = 1:numA
    sz = sizesA{i};
    cellA{i} = randn(sz);
    cellY{i} = randn(sz(1), numY);
end

%load SL0testdata
tic
for i = 1:numA
    for j = 1:numY
        cellXr{i}(:,j) = SL0(cellA{i}, cellY{i}(:,j), sigmamin(i));
    end
end
toc

save SL0testdata