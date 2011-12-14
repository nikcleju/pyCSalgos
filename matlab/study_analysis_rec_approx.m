% File: study_analysis_rec_algos
% Run experiment to prove that our approx. ABS approach really converges to
% the anaysis solution as lambda -> infty
% For this, we need to generate signals that, provably, yield bad results
% with synthesis recovery, and good results with analysis recovery
% To do this we choose, N >> d, small l, and m close to d

clear all
close all

% =================================
% Set up experiment parameters
%==================================
% Which form factor, delta and rho we want
sigma = 2;
%delta = 0.995;
%rho   = 0.7;
%delta = 0.8;
%rho = 0.15;
delta = 0.5;
rho = 0.05;

% Number of vectors to generate each time
numvects = 10;

% Add noise 
% This is norm(signal)/norm(noise), so power, not energy
SNRdb = 20;
%epsextrafactor = 2

% =================================
% Processing the parameters
%==================================

% Compute noiselevel from db
noiselevel = 1 / (10^(SNRdb/10));

% Set up parameter structure
% and generate data X as well
d = 50;
p = round(sigma*d);
m = round(delta*d);
l = round(d - rho*m);
            
% Generate Omega and data based on parameters
Omega = Generate_Analysis_Operator(d, p);
%Make Omega more coherent
[U, S, V] = svd(Omega);
%Sdnew = diag(S) ./ (1:numel(diag(S)))';
Sdnew = diag(S) .* (1:numel(diag(S)))'; % Make D coherent, not Omega!
Snew = [diag(Sdnew); zeros(size(S,1) - size(S,2), size(S,2))];
Omega = U * Snew * V';
[x0,y,M,Lambda, realnoise] = Generate_Data_Known_Omega(Omega, d,p,m,l,noiselevel, numvects,'l0');

datafilename = 'mat/data_SNR20';
save(datafilename);
%load mat/data_SNR20

% Values of lambda
%lambdas = sqrt([1e-10 1e-8 1e-6 1e-4 1e-2 1 100 1e4 1e6 1e8 1e10]);
lambdas = [0 10.^linspace(-5, 4, 10)];
%lambdas = 1000
%lambdas = sqrt([0 1 10000]);

% Algorithm identifiers
algonone = [];
ompk   = 1;
ompeps = 2;
tst    = 3;
bp     = 4;
gap    = 5;
nesta  = 6;
sl0    = 7;
yall1  = 8;
spgl1  = 9;
nestasynth = 10;
numallalgos = 10;
algoname{ompk} = 'ABS OMPk';
algoname{ompeps} = 'ABS OMPeps';
algoname{tst} = 'ABS TST';
algoname{bp} = 'ABS BP';
algoname{gap} = 'GAP';
algoname{nesta} = 'NESTA Analysis';
algoname{sl0} = 'ABS SL0';
algoname{yall1} = 'ABS YALL1';
algoname{spgl1} = 'ABS SPGL1';
algoname{nestasynth} = 'NESTA Synth';

% What algos to run
%algos = [gap, ompk, ompeps, tst, bp, sl0, yall1];
%algos = [gap, ompk, ompeps, tst, bp];
algos = [gap, sl0];
numalgos = numel(algos);

% Save mat file?
do_save_mat = 0;
matfilename = 'mat/approx_d50_sigma2_SNR20db_Dcoh';

% Save figures?
do_save_figs = 0;
figfilename = 'figs/approx_d50_sigma2_SNR20db_Dcoh';

% Show progressbar ? (not recommended when running on parallel threads)
do_progressbar = 0;
if do_progressbar
    progressbar('Total', 'Current parameters');
end

% Init times
for i = 1:numalgos
    elapsed(algos(i)) = 0;
end

% Init results structure
results = [];


% ========
% Run
% ========

% Run GAP and NESTA first
if any(algos == gap)
    for iy = 1:size(y,2)
        a = gap;
        % Compute epsilon 
        %epsilon = epsextrafactor * noiselevel * norm(y(:,iy));
        epsilon = 1.1 * norm(realnoise(:,iy));
        %
        gapparams = [];
        gapparams.num_iteration = 1000;
        gapparams.greedy_level = 0.9;
        gapparams.stopping_coefficient_size = 1e-4;
        gapparams.l2solver = 'pseudoinverse';
        %gapparams.noise_level = noiselevel;
        gapparams.noise_level = epsilon;
        timer(a) = tic;
        xrec{a}(:,iy) = GAP(y(:,iy), M, M', Omega, Omega', gapparams, zeros(d,1));
        elapsed(a) = elapsed(a) + toc(timer(a));
        %
        err{a}(iy)    = norm(x0(:,iy) - xrec{a}(:,iy));
        relerr{a}(iy) = err{a}(iy) / norm(x0(:,iy));    
    end
    disp([ algoname{a} ':   avg relative error = ' num2str(mean(relerr{a}))]);
end
if any(algos == nesta)
    for iy = 1:size(y,2)
        a = nesta;
        % Compute epsilon 
        %epsilon = epsextrafactor * noiselevel * norm(y(:,iy));
        epsilon = 1.1 * norm(realnoise(:,iy));
        %
        try
            timer(a) = tic;
            %xrec{a}(:,iy) = do_nesta_DemoNonProjector(x0(:,iy), M, Omega', 0);

            % Common heuristic: delta = sqrt(m + 2*sqrt(2*m))*sigma ?????????
            [U,S,V] = svd(M,'econ');
            opts.U = Omega;
            opts.Ut = Omega';
            opts.USV.U=U;
            opts.USV.S=S;
            opts.USV.V=V;
            opts.TolVar = 1e-5;
            opts.Verbose = 0;
            xrec{a}(:,iy) = NESTA(M, [], y(:,iy), 1e-3, epsilon, opts);
            elapsed(a) = elapsed(a) + toc(timer(a));
        catch err
            disp('*****ERROR: NESTA throwed error *****');
            xrec{a}(:,iy) = zeros(size(x0(:,iy)));
        end
        %
        err{a}(iy)    = norm(x0(:,iy) - xrec{a}(:,iy));
        relerr{a}(iy) = err{a}(iy) / norm(x0(:,iy));    
    end
    disp([ algoname{a} ':   avg relative error = ' num2str(mean(relerr{a}))]);
end

for i = 1:numel(algos)
    if algos(i) == gap
        continue;
    end
    prevgamma{algos(i)} = zeros(p, numvects);
end

% Run ABS algorithms (lambda-dependent)
for iparam = 1:numel(lambdas)
    
    % Read current parameters
    lambda = lambdas(iparam);
    
    % Init stuff
    for i = 1:numel(algos)
        if algos(i) == gap || algos(i) == nesta
            continue;
        end
        xrec{algos(i)} = zeros(d, numvects);
    end
    %
    for i = 1:numel(algos)
        if algos(i) == gap || algos(i) == nesta
            continue;
        end        
        err{algos(i)} = zeros(numvects,1);
        relerr{algos(i)} = zeros(numvects,1);
    end
    
    % For every generated signal do
    for iy = 1:size(y,2)
        
        % Compute epsilon 
        %epsilon = epsextrafactor * noiselevel * norm(y(:,iy));
        epsilon = 1.1 * norm(realnoise(:,iy));
        
        %--------------------------------
        % Reconstruct (and measure delay), Compute reconstruction error
        %--------------------------------
        for i = 1:numel(algos)
            a = algos(i);
            if a == gap  || a == nesta
                continue
            end
            if a == ompk
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_OMPk_approx(y(:,iy), Omega, M, p-l, lambda);
                elapsed(a) = elapsed(a) + toc(timer(a));
            elseif a == ompeps
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_OMPeps_approx(y(:,iy), Omega, M, epsilon, lambda, Omega * x0(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));
            elseif a == tst
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_TST_approx(y(:,iy), Omega, M, epsilon, lambda, Omega * x0(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));
            elseif a == bp
                timer(a) = tic;
                [xrec{a}(:,iy), prevgamma{a}(:,iy)] = ABS_BP_approx(y(:,iy), Omega, M, epsilon, lambda, prevgamma{a}(:,iy), Omega * x0(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));
            elseif a == yall1
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_YALL1_approx(y(:,iy), Omega, M, epsilon, lambda, Omega * x0(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));                
%             elseif a == gap
%                 gapparams = [];
%                 gapparams.num_iteration = 40;
%                 gapparams.greedy_level = 0.9;
%                 gapparams.stopping_coefficient_size = 1e-4;
%                 gapparams.l2solver = 'pseudoinverse';
%                 %gapparams.noise_level = noiselevel;
%                 gapparams.noise_level = epsilon;
%                 timer(a) = tic;
%                 xrec{a}(:,iy) = GAP(y(:,iy), M, M', Omega, Omega', gapparams, zeros(d,1));
%                 elapsed(a) = elapsed(a) + toc(timer(a));
%             elseif a == nesta
%                 try
%                     timer(a) = tic;
%                     %xrec{a}(:,iy) = do_nesta_DemoNonProjector(x0(:,iy), M, Omega', 0);
%                     
%                     % Common heuristic: delta = sqrt(m + 2*sqrt(2*m))*sigma ?????????
%                     [U,S,V] = svd(M,'econ');
%                     opts.U = Omega;
%                     opts.Ut = Omega';
%                     opts.USV.U=U;
%                     opts.USV.S=S;
%                     opts.USV.V=V;
%                     opts.TolVar = 1e-5;
%                     opts.Verbose = 0;
%                     xrec{a}(:,iy) = NESTA(M, [], y(:,iy), 1e-3, epsilon, opts);
%                     elapsed(a) = elapsed(a) + toc(timer(a));
%                 catch err
%                     disp('*****ERROR: NESTA throwed error *****');
%                     xrec{a}(:,iy) = zeros(size(x0(:,iy)));
%                 end
            elseif a == sl0
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_SL0_approx(y(:,iy), Omega, M, epsilon, lambda);
                elapsed(a) = elapsed(a) + toc(timer(a));    
            elseif a == spgl1
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_SPGL1_approx(y(:,iy), Omega, M, epsilon, lambda, Omega * xrec{bp}(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));    
            elseif a == nestasynth
                timer(a) = tic;
                xrec{a}(:,iy) = ABS_NESTAsynth_approx(y(:,iy), Omega, M, epsilon, lambda, Omega * xrec{bp}(:,iy));
                elapsed(a) = elapsed(a) + toc(timer(a));
            else
                %error('No such algorithm!');
            end
            %
            % Compare to GAP instead!
            %x0(:,iy) = xrec{gap}(:,iy);
            %
            err{a}(iy)    = norm(x0(:,iy) - xrec{a}(:,iy));
            relerr{a}(iy) = err{a}(iy) / norm(x0(:,iy));
        end

        % Update progressbar
%         if do_progressbar
%             %frac2 = iy/numvects;
%             %frac1 = ((iparam-1) + frac2) / count;
%             if norm(frac2 - 1) < 1e-6
%                 frac2 = 0;
%             end
%             frac2 = frac2 + incr2;
%             frac1 = frac1 + incr1;
%             progressbar(frac1, frac2);
%         end
    end
    
    %--------------------------------
    % Save results in big stucture & display
    %--------------------------------
    % Save reconstructed signals
    % Save rel & abs errors
    % Display error
    results(iparam).xrec = xrec;
    results(iparam).err = err;
    results(iparam).relerr = relerr;
    %
    %disp(['Simulation no. ' num2str(iparam)]);
    disp(['Lambda = ' num2str(lambda) ':']);
    for i = 1:numalgos
        a = algos(i);
        if a == gap || a == nesta
            continue
        end
        disp([ algoname{a} ':   avg relative error = ' num2str(mean(relerr{a}))]);
    end
end

% =================================
% Save
% =================================
if do_save_mat
    save(matfilename);
    disp(['Saved to ' matfilename]);
end

% =================================
% Plot
% =================================
toplot = zeros(numel(lambdas), numalgos);

relerrs = {results.relerr};
for i = 1:numalgos
    for j = 1:numel(lambdas)
        toplot(j,i) = mean(relerrs{j}{algos(i)});
    end
end

%h = plot(toplot);
h = semilogx(lambdas, toplot);
legend(algoname{algos})
xlabel('Lambda')
ylabel('Average reconstruction error')
title('Reconstruction error with different algorithms')

if (do_save_figs)
    saveas(gcf, [figfilename '.fig']);
    saveas(gcf, [figfilename '.png']);
    saveTightFigure(gcf, [figfilename '.pdf']);    
end

