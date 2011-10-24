function beta = RecommendedTST(X,Y, nsweep,tol,xinitial,ro)

% function beta=RecommendedTST(X,y, nsweep,tol,xinitial,ro)
% This function gets the measurement matrix and the measurements and
% the number of runs and applies the TST algorithm with optimally tuned parameters
% to the problem. For more information you may refer to the paper,
% "Optimally tuned iterative reconstruction algorithms for compressed
% sensing," by Arian Maleki and David L. Donoho. 
%           X  : Measurement matrix; We assume that all the columns have
%               almost equal $\ell_2$ norms. The tunning has been done on
%               matrices with unit column norm. 
%            y : output vector
%       nsweep : number of iterations. The default value is 300.
%          tol : if the relative prediction error i.e. ||Y-Ax||_2/ ||Y||_2 <
%               tol the algorithm will stop. If not provided the default
%               value is zero and tha algorithm will run for nsweep
%               iterations. The Default value is 0.00001.
%     xinitial : This is an optional parameter. It will be used as an
%                initialization of the algorithm. All the results mentioned
%                in the paper have used initialization at the zero vector
%                which is our default value. For default value you can enter
%                0 here. 
%        ro    : This is a again an optional parameter. If not given the
%                algorithm will use the default optimal values. It specifies
%                the sparsity level. For the default value you may also used if
%                rostar=0;
%
% Outputs:
%      beta : the estimated coeffs.
%
% References:
% For more information about this algorithm and to find the papers about
% related algorithms like CoSaMP and SP please refer to the paper mentioned 
% above and the references of that paper.


colnorm=mean((sum(X.^2)).^(.5));
X=X./colnorm;
Y=Y./colnorm;
[n,p]=size(X);
delta=n/p;
if nargin<3
    nsweep=300;
end
if nargin<4
    tol=0.00001;
end
if nargin<5 | xinitial==0
    xinitial = zeros(p,1);
end
if nargin<6 | ro==0
    ro=0.044417*delta^2+ 0.34142*delta+0.14844;
end


k1=floor(ro*n);
k2=floor(ro*n);


%initialization
x1=xinitial;
I=[];

for sweep=1:nsweep
    r=Y-X*x1;
    c=X'*r;
    [csort, i_csort]=sort(abs(c));
    I=union(I,i_csort(end-k2+1:end));
    xt = X(:,I) \ Y;
    [xtsort, i_xtsort]=sort(abs(xt));

    J=I(i_xtsort(end-k1+1:end));
    x1=zeros(p,1);
    x1(J)=xt(i_xtsort(end-k1+1:end));
    I=J;
    if norm(Y-X*x1)/norm(Y)<tol
        break
    end

end

beta=x1;
