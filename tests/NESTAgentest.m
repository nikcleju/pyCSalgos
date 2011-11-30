% Run NESTA and save parameters and solutions as reference test data
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

for i = 1:numA
    sz = sizesA{i};
    cellA{i} = randn(sz);
    m = round((0.2 + 0.6*rand)*sz(2));
    cellM{i} = randn(m,sz(2));
    cellY{i} = randn(m, numY);
    %cellXinit{i} = zeros(sz(2), numY);
    for j = 1:numY
        cellEps{i}(j) = rand / 100; % restrict from 0 to 1% of measurements
    end
end
opt_TolVar = 1e-5;
opt_Verbose = 0;
opt_muf = 1e-3;
opt_l2solver = 'pseudoinverse';

%load NESTAtestdata
tic
for i = 1:numA
    [U,S,V] = svd(cellM{i},'econ');
    opts.U = cellA{i};
    opts.Ut = cellA{i}';
    opts.USV.U=U;
    opts.USV.S=S;
    opts.USV.V=V;
    opts.TolVar = opt_TolVar;
    opts.Verbose = opt_Verbose;    
    for j = 1:numY
        cellXr{i}(:,j) = NESTA(cellM{i}, [], cellY{i}(:,j), opt_muf, cellEps{i}(j) * norm(cellY{i}(:,j)), opts);
    end
    disp(['Finished sz ' num2str(i)])
end
toc

save NESTAtestdata