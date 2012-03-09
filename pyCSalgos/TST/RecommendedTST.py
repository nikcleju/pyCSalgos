
import numpy as np
import math

#function beta = RecommendedTST(X,Y, nsweep,tol,xinitial,ro)
def RecommendedTST(X, Y, nsweep=300, tol=0.00001, xinitial=None, ro=None):

  # function beta=RecommendedTST(X,y, nsweep,tol,xinitial,ro)
  # This function gets the measurement matrix and the measurements and
  # the number of runs and applies the TST algorithm with optimally tuned parameters
  # to the problem. For more information you may refer to the paper,
  # "Optimally tuned iterative reconstruction algorithms for compressed
  # sensing," by Arian Maleki and David L. Donoho. 
  #           X  : Measurement matrix; We assume that all the columns have
  #               almost equal $\ell_2$ norms. The tunning has been done on
  #               matrices with unit column norm. 
  #            y : output vector
  #       nsweep : number of iterations. The default value is 300.
  #          tol : if the relative prediction error i.e. ||Y-Ax||_2/ ||Y||_2 <
  #               tol the algorithm will stop. If not provided the default
  #               value is zero and tha algorithm will run for nsweep
  #               iterations. The Default value is 0.00001.
  #     xinitial : This is an optional parameter. It will be used as an
  #                initialization of the algorithm. All the results mentioned
  #                in the paper have used initialization at the zero vector
  #                which is our default value. For default value you can enter
  #                0 here. 
  #        ro    : This is a again an optional parameter. If not given the
  #                algorithm will use the default optimal values. It specifies
  #                the sparsity level. For the default value you may also used if
  #                rostar=0;
  #
  # Outputs:
  #      beta : the estimated coeffs.
  #
  # References:
  # For more information about this algorithm and to find the papers about
  # related algorithms like CoSaMP and SP please refer to the paper mentioned 
  # above and the references of that paper.
  #
  #---------------------
  # Original Matab code:
  #        
  #colnorm=mean((sum(X.^2)).^(.5));
  #X=X./colnorm;
  #Y=Y./colnorm;
  #[n,p]=size(X);
  #delta=n/p;
  #if nargin<3
  #    nsweep=300;
  #end
  #if nargin<4
  #    tol=0.00001;
  #end
  #if nargin<5 | xinitial==0
  #    xinitial = zeros(p,1);
  #end
  #if nargin<6 | ro==0
  #    ro=0.044417*delta^2+ 0.34142*delta+0.14844;
  #end
  #
  #
  #k1=floor(ro*n);
  #k2=floor(ro*n);
  #
  #
  ##initialization
  #x1=xinitial;
  #I=[];
  #
  #for sweep=1:nsweep
  #    r=Y-X*x1;
  #    c=X'*r;
  #    [csort, i_csort]=sort(abs(c));
  #    I=union(I,i_csort(end-k2+1:end));
  #    xt = X(:,I) \ Y;
  #    [xtsort, i_xtsort]=sort(abs(xt));
  #
  #    J=I(i_xtsort(end-k1+1:end));
  #    x1=zeros(p,1);
  #    x1(J)=xt(i_xtsort(end-k1+1:end));
  #    I=J;
  #    if norm(Y-X*x1)/norm(Y)<tol
  #        break
  #    end
  #
  #end
  #
  #beta=x1;
  #
  # End of original Matab code
  #----------------------------
  
  #colnorm=mean((sum(X.^2)).^(.5));
  colnorm = np.mean(np.sqrt((X**2).sum(0)))
  X = X / colnorm
  Y = Y / colnorm
  [n,p] = X.shape
  delta = float(n) / p
  #  if nargin<3
  #      nsweep=300;
  #  end
  #  if nargin<4
  #      tol=0.00001;
  #  end
  #  if nargin<5 | xinitial==0
  #      xinitial = zeros(p,1);
  #  end
  #  if nargin<6 | ro==0
  #      ro=0.044417*delta^2+ 0.34142*delta+0.14844;
  #  end
  if xinitial is None:
    xinitial = np.zeros(p)
  if ro == None:
    ro = 0.044417*delta**2 + 0.34142*delta + 0.14844
  
  k1 = math.floor(ro*n)
  k2 = math.floor(ro*n)
  
  #initialization
  x1 = xinitial.copy()
  I = []
  
  for sweep in np.arange(nsweep):
      r = Y - np.dot(X,x1)
      c = np.dot(X.T, r)
      #[csort, i_csort] = np.sort(np.abs(c))
      i_csort = np.argsort(np.abs(c))
      #I = numpy.union1d(I , i_csort(end-k2+1:end))
      I = np.union1d(I , i_csort[-k2:])
      #xt = X[:,I] \ Y
      # Make sure X[:,np.int_(I)] is a 2-dimensional matrix even if I has a single value (and therefore yields a column)
      if I.size is 1:
        a = np.reshape(X[:,np.int_(I)],(X.shape[0],1))
      else:
        a = X[:,np.int_(I)]
      xt = np.linalg.lstsq(a, Y)[0]
      #[xtsort, i_xtsort] = np.sort(np.abs(xt))
      i_xtsort = np.argsort(np.abs(xt))
  
      J = I[i_xtsort[-k1:]]
      x1 = np.zeros(p)
      x1[np.int_(J)] = xt[i_xtsort[-k1:]]
      I = J.copy()
      if np.linalg.norm(Y-np.dot(X,x1)) / np.linalg.norm(Y) < tol:
          break
      #end
      
  #end
  
  return x1.copy()

