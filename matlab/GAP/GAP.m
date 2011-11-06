function [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, xinit)

%%
% [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, xinit)
%
% Greedy Analysis Pursuit Algorithm
% This aims to find an approximate (sometimes exact) solution of
%    xhat = argmin || Omega * x ||_0   subject to   || y - M * x ||_2 <= epsilon.
%
% Outputs:
%   xhat : estimate of the target cosparse vector x0.
%   Lambdahat : estimate of the cosupport of x0.
%
% Inputs:
%   y : observation/measurement vector of a target cosparse solution x0,
%       given by relation  y = M * x0 + noise.
%   M : measurement matrix. This should be given either as a matrix or as a function handle
%       which implements linear transformation.
%   MH : conjugate transpose of M. 
%   Omega : analysis operator. Like M, this should be given either as a matrix or as a function
%           handle which implements linear transformation.
%   OmegaH : conjugate transpose of OmegaH.
%   params : parameters that govern the behavior of the algorithm (mostly).
%      params.num_iteration : GAP performs this number of iterations.
%      params.greedy_level : determines how many rows of Omega GAP eliminates at each iteration.
%                            if the value is < 1, then the rows to be eliminated are determined by
%                                j : |omega_j * xhat| > greedy_level * max_i |omega_i * xhat|.
%                            if the value is >= 1, then greedy_level is the number of rows to be
%                            eliminated at each iteration.
%      params.stopping_coefficient_size : when the maximum analysis coefficient is smaller than
%                                         this, GAP terminates.
%      params.l2solver : legitimate values are 'pseudoinverse' or 'cg'. determines which method
%                        is used to compute
%                        argmin || Omega_Lambdahat * x ||_2   subject to  || y - M * x ||_2 <= epsilon.
%      params.l2_accuracy : when l2solver is 'cg', this determines how accurately the above 
%                           problem is solved.
%      params.noise_level : this corresponds to epsilon above.
%   xinit : initial estimate of x0 that GAP will start with. can be zeros(d, 1).
%
% Examples:
%
% Not particularly interesting:
% >> d = 100; p = 110; m = 60; 
% >> M = randn(m, d);
% >> Omega = randn(p, d);
% >> y = M * x0 + noise;
% >> params.num_iteration = 40;
% >> params.greedy_level = 0.9;
% >> params.stopping_coefficient_size = 1e-4;
% >> params.l2solver = 'pseudoinverse';
% >> [xhat, Lambdahat] = GAP(y, M, M', Omega, Omega', params, zeros(d, 1));
%
% Assuming that FourierSampling.m, FourierSamplingH.m, FDAnalysis.m, etc. exist:
% >> n = 128;
% >> M = @(t) FourierSampling(t, n);
% >> MH = @(u) FourierSamplingH(u, n);
% >> Omega = @(t) FDAnalysis(t, n);
% >> OmegaH = @(u) FDSynthesis(t, n);
% >> params.num_iteration = 1000;
% >> params.greedy_level = 50;
% >> params.stopping_coefficient_size = 1e-5;
% >> params.l2solver = 'cg';   % in fact, 'pseudoinverse' does not even make sense.
% >> [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, zeros(d, 1));
%
% Above: FourierSampling and FourierSamplingH are conjugate transpose of each other.
%        FDAnalysis and FDSynthesis are conjugate transpose of each other.
%        These routines are problem specific and need to be implemented by the user.

d = length(xinit(:));

if strcmp(class(Omega), 'function_handle')
    p = length(Omega(zeros(d,1)));
else    %% Omega is a matrix
    p = size(Omega, 1);
end

iter = 0;
lagmult = 1e-4;
Lambdahat = 1:p;
while iter < params.num_iteration
    iter = iter + 1;
    [xhat, analysis_repr, lagmult] = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, lagmult, params);
    [to_be_removed, maxcoef] = FindRowsToRemove(analysis_repr, params.greedy_level);
    %disp(['** maxcoef=', num2str(maxcoef), ' target=', num2str(params.stopping_coefficient_size), ' rows_remaining=', num2str(length(Lambdahat)), ' lagmult=', num2str(lagmult)]);
    if check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params)
        break;
    end
    xinit = xhat;
    Lambdahat(to_be_removed) = [];

    %n = sqrt(d);
    %figure(9);
    %RR = zeros(2*n, n-1);
    %RR(Lambdahat) = 1;
    %XD = ones(n, n);
    %XD(:, 2:end) = XD(:, 2:end) .* RR(1:n, :);
    %XD(:, 1:(end-1)) = XD(:, 1:(end-1)) .* RR(1:n, :);
    %XD(2:end, :) = XD(2:end, :) .* RR((n+1):(2*n), :)';
    %XD(1:(end-1), :) = XD(1:(end-1), :) .* RR((n+1):(2*n), :)';
    %XD = FD2DiagnosisPlot(n, Lambdahat);
    %imshow(XD);
    %figure(10);
    %imshow(reshape(real(xhat), n, n));
end
return;


function [to_be_removed, maxcoef] = FindRowsToRemove(analysis_repr, greedy_level)

    abscoef = abs(analysis_repr(:));
    n = length(abscoef);
    maxcoef = max(abscoef);
    if greedy_level >= 1
        qq = quantile(abscoef, 1-greedy_level/n);
    else
        qq = maxcoef*greedy_level;
    end

    to_be_removed = find(abscoef >= qq);
    return;

function r = check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params)

    r = 0;

    if isfield(params, 'stopping_coefficient_size') && maxcoef < params.stopping_coefficient_size
        r = 1;
        return;
    end

    if isfield(params, 'stopping_lagrange_multiplier_size') && lagmult > params.stopping_lagrange_multiplier_size
        r = 1;
        return;
    end

    if isfield(params, 'stopping_relative_solution_change') && norm(xhat-xinit)/norm(xhat) < params.stopping_relative_solution_change
        r = 1;
        return;
    end

    if isfield(params, 'stopping_cosparsity') && length(Lambdahat) < params.stopping_cosparsity
        r = 1;
        return;
    end
