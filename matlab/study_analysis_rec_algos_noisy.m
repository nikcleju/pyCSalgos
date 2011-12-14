% File: study_analysis_rec_algos
% Run global experiment to compare algorithms used for analysis-based reconstruction
% and plot phast transition graphs

clear all
close all

% =================================
% Set up experiment parameters
%==================================
% Which form factor, delta and rho we want
sigmas = 1.2;
deltas = 0.05:0.05:0.95;
rhos   = 0.05:0.05:0.95;
% deltas = [0.95];
% rhos   = [0.1];
%deltas = 0.3:0.3:0.9;
%rhos   = 0.3:0.3:0.9;

% Number of vectors to generate each time
numvects = 100;

% Add noise 
% This is norm(signal)/norm(noise), so power, not energy
SNRdb = 20; % virtually no noise

% Show progressbar ? (not recommended when running on parallel threads)
do_progressbar = 0;

% Value of lambda
lambda = 1e-2;

% What algos to run
do_abs_ompk = 1;
do_abs_ompeps = 1;
do_abs_tst = 1;
do_abs_bp = 1;
do_gap = 1;
do_nesta = 0;

% =================================
% Processing the parameters
%==================================
% Set up parameter structure
count = 0;
for isigma = 1:sigmas
    for idelta = 1:numel(deltas)
        for irho = 1:numel(rhos)
            sigma = sigmas(isigma);
            delta = deltas(idelta);
            rho = rhos(irho);
        
            d = 200;
            p = round(sigma*d);
            m = round(delta*d);
            l = round(d - rho*m);
            
            params(count+1).d = d;
            params(count+1).p = p;
            params(count+1).m = m;
            params(count+1).l = l;
            
            count = count + 1;
        end
    end
end

% Compute noiselevel from db
noiselevel = 1 / (10^(SNRdb/10));

%load study_analysis_init

% Generate an analysis operator Omega
Omega = Generate_Analysis_Operator(d, p);

% Progressbar
if do_progressbar
    progressbar('Total', 'Current parameters');
end

% Init times
elapsed_abs_ompk = 0;
elapsed_abs_ompeps = 0;
elapsed_abs_tst = 0;
elapsed_abs_bp = 0;
elapsed_gap = 0;
elapsed_nesta = 0;

% Init results structure
results = [];

% Prepare progressbar reduction variables
% if do_progressbar
%     incr2 = 1/numvects;
%     incr1 = 1/numvects/count;
%     frac2 = 0;
%     frac1 = 0;
% end 

% ========
% Run
% ========
parfor iparam = 1:numel(params)
    
    % Read current parameters
    d = params(iparam).d;
    p = params(iparam).p;
    m = params(iparam).m;
    l = params(iparam).l;
    
    % Init stuff
    xrec_abs_ompk   = zeros(d, numvects);
    xrec_abs_ompeps = zeros(d, numvects);
    xrec_abs_tst    = zeros(d, numvects);
    xrec_abs_bp     = zeros(d, numvects);
    xrec_gap        = zeros(d, numvects);
    xrec_nesta      = zeros(d, numvects);
    %
       err_abs_ompk   = zeros(numvects,1);
    relerr_abs_ompk   = zeros(numvects,1);
       err_abs_ompeps = zeros(numvects,1);
    relerr_abs_ompeps = zeros(numvects,1);
       err_abs_tst    = zeros(numvects,1);
    relerr_abs_tst    = zeros(numvects,1);
       err_abs_bp     = zeros(numvects,1);
    relerr_abs_bp     = zeros(numvects,1);
       err_gap        = zeros(numvects,1);
    relerr_gap        = zeros(numvects,1);
       err_nesta      = zeros(numvects,1);
    relerr_nesta      = zeros(numvects,1);

    % Generate data based on parameters
    [x0,y,M,Lambda] = Generate_Data_Known_Omega(Omega, d,p,m,l,noiselevel, numvects,'l0');
    
    % For every generated signal do
    for iy = 1:size(y,2)
        
        % Compute epsilon 
        epsilon = noiselevel * norm(y(:,iy));
        
        %--------------------------------
        % Reconstruct (and measure delay)
        % Compute reconstruction error
        %--------------------------------
        % ABS-OMPk
        if (do_abs_ompk)
            timer_abs_ompk = tic;
            xrec_abs_ompk(:,iy) = ABS_OMPk_approx(y(:,iy), Omega, M, p-l, lambda);
            elapsed_abs_ompk = elapsed_abs_ompk + toc(timer_abs_ompk);
            %
            err_abs_ompk(iy)    = norm(x0(:,iy) - xrec_abs_ompk(:,iy));
            relerr_abs_ompk(iy) = err_abs_ompk(iy) / norm(x0(:,iy));   
        end
        % ABS-OMPeps
        if (do_abs_ompeps)
            timer_abs_ompeps = tic;
            xrec_abs_ompeps(:,iy) = ABS_OMPeps_approx(y(:,iy), Omega, M, epsilon, lambda);
            elapsed_abs_ompeps = elapsed_abs_ompeps + toc(timer_abs_ompeps);
            %
            err_abs_ompeps(iy)    = norm(x0(:,iy) - xrec_abs_ompeps(:,iy));
            relerr_abs_ompeps(iy) = err_abs_ompeps(iy) / norm(x0(:,iy));
        end
        % ABS-TST
        if (do_abs_tst)
            timer_abs_tst = tic;
            xrec_abs_tst(:,iy) = ABS_TST_approx(y(:,iy), Omega, M, epsilon, lambda);
            elapsed_abs_tst = elapsed_abs_tst + toc(timer_abs_tst);
            %
            err_abs_tst(iy)     = norm(x0(:,iy) - xrec_abs_tst(:,iy));
            relerr_abs_tst(iy)  = err_abs_tst(iy) / norm(x0(:,iy));
        end
        % ABS-BP
        if (do_abs_bp)
            timer_abs_bp = tic;
            xrec_abs_bp(:,iy)  = ABS_BP_approx(y(:,iy), Omega, M, epsilon, lambda);
            elapsed_abs_bp = elapsed_abs_bp + toc(timer_abs_bp);
            %
            err_abs_bp(iy)     = norm(x0(:,iy) - xrec_abs_bp(:,iy));
            relerr_abs_bp(iy)  = err_abs_bp(iy) / norm(x0(:,iy));
        end
        % GAP
        if (do_gap)
            gapparams = [];
            gapparams.num_iteration = 40;
            gapparams.greedy_level = 0.9;
            gapparams.stopping_coefficient_size = 1e-4;
            gapparams.l2solver = 'pseudoinverse';
            gapparams.noise_level = noiselevel;
            timer_gap = tic;
            xrec_gap(:,iy) = GAP(y(:,iy), M, M', Omega, Omega', gapparams, zeros(d,1));
            elapsed_gap = elapsed_gap + toc(timer_gap);
            %
            err_gap(iy)     = norm(x0(:,iy) - xrec_gap(:,iy));
            relerr_gap(iy)  = err_gap(iy) / norm(x0(:,iy));
        end
        % NESTA
        if (do_nesta)
            try
                timer_nesta = tic;
                xrec_nesta(:,iy) = do_nesta_DemoNonProjector(x0(:,iy), M, Omega', 0);
                elapsed_nesta = elapsed_nesta + toc(timer_nesta);
            catch err
                disp('*****ERROR: NESTA throwed error *****');
                xrec_nesta(:,iy) = zeros(size(x0(:,iy)));
            end
            %
            err_nesta(iy)       = norm(x0(:,iy) - xrec_nesta(:,iy));
            relerr_nesta(iy)    = err_nesta(iy) / norm(x0(:,iy)); 
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
    disp(['Simulation no. ' num2str(iparam)]);
    if (do_abs_ompk)
        results(iparam).xrec_abs_ompk   = xrec_abs_ompk;
        results(iparam).err_abs_ompk    = err_abs_ompk;
        results(iparam).relerr_abs_ompk = relerr_abs_ompk;
        disp(['  ABS_OMPk:   avg relative error = ' num2str(mean(relerr_abs_ompk))]);
    end
    if (do_abs_ompeps)
        results(iparam).xrec_abs_ompeps   = xrec_abs_ompeps;
        results(iparam).err_abs_ompeps    = err_abs_ompeps;
        results(iparam).relerr_abs_ompeps = relerr_abs_ompeps;   
        disp(['  ABS_OMPeps: avg relative error = ' num2str(mean(relerr_abs_ompeps))]);
    end
    if (do_abs_tst)
        results(iparam).xrec_abs_tst   = xrec_abs_tst;
        results(iparam).err_abs_tst    = err_abs_tst;
        results(iparam).relerr_abs_tst = relerr_abs_tst;
        disp(['  ABS_TST:    avg relative error = ' num2str(mean(relerr_abs_tst))]);
    end
    if (do_abs_bp)
        results(iparam).xrec_abs_bp   = xrec_abs_bp;
        results(iparam).err_abs_bp    = err_abs_bp;
        results(iparam).relerr_abs_bp = relerr_abs_bp;
        disp(['  ABS_BP:     avg relative error = ' num2str(mean(relerr_abs_bp))]);
    end
    if (do_gap)
        results(iparam).xrec_gap   = xrec_gap;
        results(iparam).err_gap    = err_gap;
        results(iparam).relerr_gap = relerr_gap;
        disp(['  GAP:        avg relative error = ' num2str(mean(relerr_gap))]);
    end
    if (do_nesta)
        results(iparam).xrec_nesta   = xrec_nesta;
        results(iparam).err_nesta    = err_nesta;
        results(iparam).relerr_nesta = relerr_nesta;
        disp(['  NESTA:      avg relative error = ' num2str(mean(relerr_nesta))]);
    end
end

% =================================
% Save
% =================================
save mat/avgerr_SNR20_lbd1e-2

% =================================
% Plot phase transition
% =================================
%--------------------------------
% Prepare
%--------------------------------
%d = 200;
%m = 190;
%exactthr = d/m * noiselevel;
%sigma = 1.2;
iparam = 1;
for idelta = 1:numel(deltas)
    for irho = 1:numel(rhos)
        % Create exact recovery count matrix 
%         nexact_abs_bp  (irho, idelta)    = sum(results(iparam).relerr_abs_bp < exactthr);
%         nexact_abs_ompk (irho, idelta)   = sum(results(iparam).relerr_abs_ompk < exactthr);
%         nexact_abs_ompeps (irho, idelta) = sum(results(iparam).relerr_abs_ompeps < exactthr);
%         nexact_gap (irho, idelta)        = sum(results(iparam).relerr_gap < exactthr);
%         nexact_abs_tst (irho, idelta)    = sum(results(iparam).relerr_abs_tst < exactthr);
% %         nexact_nesta(irho, idelta)       = sum(results(iparam).relerr_nesta < exactthr);

        % Get histogram (for a single param set only!)
%         hist_abs_ompk   = hist(results(iparam).relerr_abs_ompk, 0.001:0.001:0.1);
%         hist_abs_ompeps = hist(results(iparam).relerr_abs_ompeps, 0.001:0.001:0.1);
%         hist_abs_tst    = hist(results(iparam).relerr_abs_tst, 0.001:0.001:0.1);
%         hist_abs_bp     = hist(results(iparam).relerr_abs_bp, 0.001:0.001:0.1);
%         hist_gap        = hist(results(iparam).relerr_gap, 0.001:0.001:0.1);
        
        % Compute average error value
        if (do_abs_ompk)
            avgerr_abs_ompk(irho, idelta)    = 1 - mean(results(iparam).relerr_abs_ompk);
            avgerr_abs_ompk(avgerr_abs_ompk < 0) = 0;
        end
        if (do_abs_ompeps)
            avgerr_abs_ompeps(irho, idelta)  = 1 - mean(results(iparam).relerr_abs_ompeps);
            avgerr_abs_ompeps(avgerr_abs_ompeps < 0) = 0;
        end
        if (do_abs_tst)
            avgerr_abs_tst(irho, idelta)     = 1 - mean(results(iparam).relerr_abs_tst);
            avgerr_abs_tst(avgerr_abs_tst < 0) = 0;
        end
        if (do_abs_bp)
            avgerr_abs_bp(irho, idelta)      = 1 - mean(results(iparam).relerr_abs_bp);
            avgerr_abs_bp(avgerr_abs_bp < 0) = 0;
        end
        if (do_gap)
            avgerr_gap(irho, idelta)         = 1 - mean(results(iparam).relerr_gap);
            avgerr_gap(avgerr_gap < 0) = 0;
        end
        if (do_nesta)
            avgerr_nesta(irho, idelta)       = 1 - mean(results(iparam).relerr_nesta);
            avgerr_nesta(avgerr_nesta < 0) = 0;
        end
        
        iparam = iparam + 1;
    end
end

%--------------------------------
% Plot
%--------------------------------
show_phasetrans = @show_phasetrans_win;
iptsetpref('ImshowAxesVisible', 'on');
close all
figbase = 'figs/avgerr_SNR20_lbd1e-2_';
do_save = 1;
%
if (do_abs_ompk)
    figure;
    %h = show_phasetrans(nexact_abs_ompk, numvects);
    %bar(0.001:0.001:0.1, hist_abs_ompk);
    h = show_phasetrans(avgerr_abs_ompk, 1);
    if do_save
        figname = [figbase 'ABS_OMPk'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end
%
if (do_abs_ompeps)
    figure;
    %h = show_phasetrans(nexact_abs_ompeps, numvects);
    %bar(0.001:0.001:0.1, hist_abs_ompeps);
    h = show_phasetrans(avgerr_abs_ompeps, 1);
    if do_save
        figname = [figbase 'ABS_OMPeps'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end
%
if (do_abs_tst)
    figure;
    %h = show_phasetrans(nexact_abs_tst, numvects);
    %bar(0.001:0.001:0.1, hist_abs_tst);
    h = show_phasetrans(avgerr_abs_tst, 1);
    if do_save
        figname = [figbase 'ABS_TST'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end
%
if (do_abs_bp)
    figure;
    %h = show_phasetrans(nexact_abs_bp, numvects);
    %bar(0.001:0.001:0.1, hist_abs_bp);
    h = show_phasetrans(avgerr_abs_bp, 1);
    if do_save
        figname = [figbase 'ABS_BP'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end
%
if (do_gap)
    figure;
    %h = show_phasetrans(nexact_gap, numvects);
    %bar(0.001:0.001:0.1, hist_gap);
    h = show_phasetrans(avgerr_gap, 1);
    if do_save
        figname = [figbase 'GAP'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end
%
if (do_nesta)
    figure;
    %h = show_phasetrans(nexact_nesta, numvects);
    %bar(0.001:0.001:0.1, hist_nesta);
    h = show_phasetrans(avgerr_nesta, 1);
    if do_save
        figname = [figbase 'NESTA'];
        saveas(h, [figname '.fig']);
        saveas(h, [figname '.png']);
        saveTightFigure(h, [figname '.pdf']);
    end
end