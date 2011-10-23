% Run BP and save parameters and solutions as reference test data
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
        Eps{i}(j) = rand / 5; % restrict from 0 to 20% if measurements
        X0{i}(:,j) = cellA{i} \ cellY{i}(:,j);
    end
    %X0{i} = zeros(sz(1), numY);
    
end

for i = 1:numA
    for j = 1:numY
        Xr{i}(:,j) = l1qc_logbarrier_quiet(X0{i}(:,j), cellA{i}, [], cellY{i}(:,j), Eps{i}(j));
    end
end

save BPtestdata