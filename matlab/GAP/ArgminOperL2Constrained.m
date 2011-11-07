function [xhat, arepr, lagmult] = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, ilagmult, params)

%
% This function aims to compute
%    xhat = argmin || Omega(Lambdahat, :) * x ||_2   subject to  || y - M*x ||_2 <= epsilon.
% arepr is the analysis representation corresponding to Lambdahat, i.e.,
%    arepr = Omega(Lambdahat, :) * xhat.
% The function also returns the lagrange multiplier in the process used to compute xhat.
%
% Inputs:
%    y : observation/measurements of an unknown vector x0. It is equal to M*x0 + noise.
%    M : Measurement matrix
%    MH : M', the conjugate transpose of M
%    Omega : analysis operator
%    OmegaH : Omega', the conjugate transpose of Omega. Also, synthesis operator.
%    Lambdahat : an index set indicating some rows of Omega.
%    xinit : initial estimate that will be used for the conjugate gradient algorithm.
%    ilagmult : initial lagrange multiplier to be used in
%    params : parameters
%        params.noise_level : this corresponds to epsilon above.
%        params.max_inner_iteration : `maximum' number of iterations in conjugate gradient method.
%        params.l2_accurary : the l2 accuracy parameter used in conjugate gradient method
%        params.l2solver : if the value is 'pseudoinverse', then direct matrix computation (not conjugate gradient method) is used. Otherwise, conjugate gradient method is used.
%

d = length(xinit);
lagmultmax = 1e5;
lagmultmin = 1e-4;
lagmultfactor = 2;
accuracy_adjustment_exponent = 4/5;
lagmult = max(min(ilagmult, lagmultmax), lagmultmin);
was_infeasible = 0;
was_feasible = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation done using direct matrix computation from matlab. (no conjugate gradient method.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(params.l2solver, 'pseudoinverse')
    if strcmp(class(M), 'double') && strcmp(class(Omega), 'double')
        while true
            alpha = sqrt(lagmult);
            xhat = [M; alpha*Omega(Lambdahat,:)]\[y; zeros(length(Lambdahat), 1)];
            temp = norm(y - M*xhat, 2);
            %disp(['fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult)]);
            if temp <= params.noise_level
                was_feasible = 1;
                if was_infeasible == 1
                    break;
                else
                    lagmult = lagmult*lagmultfactor;
                end
            elseif temp > params.noise_level
                was_infeasible = 1;
                if was_feasible == 1
                    xhat = xprev;
                    break;
                end
                lagmult = lagmult/lagmultfactor;
            end
            if lagmult < lagmultmin || lagmult > lagmultmax
                break;
            end
            xprev = xhat;
        end
        arepr = Omega(Lambdahat, :) * xhat;
        return;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation using conjugate gradient method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(class(MH),'function_handle') 
    b = MH(y);
else
    b = MH * y;
end
norm_b = norm(b, 2);
xhat = xinit;
xprev = xinit;
residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
direction = -residual;
iter = 0;

while iter < params.max_inner_iteration
    iter = iter + 1;
    alpha = norm(residual,2)^2 / (direction' * TheHermitianMatrix(direction, M, MH, Omega, OmegaH, Lambdahat, lagmult));
    xhat = xhat + alpha*direction;
    prev_residual = residual;
    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
    beta = norm(residual,2)^2 / norm(prev_residual,2)^2;
    direction = -residual + beta*direction;

    if norm(residual,2)/norm_b < params.l2_accuracy*(lagmult^(accuracy_adjustment_exponent)) || iter == params.max_inner_iteration
        if strcmp(class(M), 'function_handle')
            temp = norm(y-M(xhat), 2);
        else
            temp = norm(y-M*xhat, 2);
        end

        if strcmp(class(Omega), 'function_handle')
            u = Omega(xhat);
            u = sqrt(lagmult)*norm(u(Lambdahat), 2);
        else
            u = sqrt(lagmult)*norm(Omega(Lambdahat,:)*xhat, 2);
        end

        %disp(['residual=', num2str(norm(residual,2)), ' norm_b=', num2str(norm_b), ' omegapart=', num2str(u), ' fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult), ' iter=', num2str(iter)]);

        if temp <= params.noise_level
            was_feasible = 1;
            if was_infeasible == 1
                break;
            else
                lagmult = lagmultfactor*lagmult;
                residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
                direction = -residual;
                iter = 0;
            end
        elseif temp > params.noise_level
            lagmult = lagmult/lagmultfactor;
            if was_feasible == 1
                xhat = xprev;
                break;
            end
            was_infeasible = 1;
            residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
            direction = -residual;
            iter = 0;
        end
        if lagmult > lagmultmax || lagmult < lagmultmin
            break;
        end
        xprev = xhat;
    %elseif norm(xprev-xhat)/norm(xhat) < 1e-2
    %    disp(['rel_change=', num2str(norm(xprev-xhat)/norm(xhat))]);
    %    if strcmp(class(M), 'function_handle')
    %        temp = norm(y-M(xhat), 2);
    %    else
    %        temp = norm(y-M*xhat, 2);
    %    end
%
%        if temp > 1.2*params.noise_level
%            was_infeasible = 1;
%            lagmult = lagmult/lagmultfactor;
%            xprev = xhat;
%        end
    end

end
disp(['fidelity_error=', num2str(temp)]);
if iter == params.max_inner_iteration
    %disp('max_inner_iteration reached. l2_accuracy not achieved.');
end

%%
% Compute analysis representation for xhat
%%
if strcmp(class(Omega),'function_handle') 
    temp = Omega(xhat);
    arepr = temp(Lambdahat);
else    %% here Omega is assumed to be a matrix
    arepr = Omega(Lambdahat, :) * xhat;
end
return;


%%
% This function computes (M'*M + lm*Omega(L,:)'*Omega(L,:)) * x.
%%
function w = TheHermitianMatrix(x, M, MH, Omega, OmegaH, L, lm)
    if strcmp(class(M), 'function_handle')
        w = MH(M(x));
    else    %% M and MH are matrices
        w = MH * M * x;
    end
    if strcmp(class(Omega),'function_handle')
        v = Omega(x);
        vt = zeros(size(v));
        vt(L) = v(L);
        w = w + lm*OmegaH(vt);
    else    %% Omega is assumed to be a matrix and OmegaH is its conjugate transpose
        w = w + lm*OmegaH(:, L)*Omega(L, :)*x;
    end
