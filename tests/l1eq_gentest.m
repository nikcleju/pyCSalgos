% Run l1eq_pd() and save parameters and solutions as reference test data
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

for i = 1:numA
    sz = sizesA{i};
    cellA{i} = randn(sz);
    cellY{i} = randn(sz(1), numY);
    for j = 1:numY
        X0{i}(:,j) = cellA{i} \ cellY{i}(:,j);
    end
    
end

for i = 1:numA
    for j = 1:numY
        Xr{i}(:,j) = l1eq_pd(X0{i}(:,j), cellA{i}, [], cellY{i}(:,j));
    end
end

save l1eq_testdata