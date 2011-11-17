# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:47:36 2011

Solve l1 minimization with quadratic AND equality constraints

@author: ncleju
"""


import numpy as np
import scipy.linalg
import math

class l1qecInputValueError(Exception):
  pass

# This is not normally used, so it is not tested, probably doesn't work
def cgsolve(A, b, tol, maxiter, verbose=1):
    raise Exception('Shouldn\'t run cgsolve(), as this is absolutely not tested. Comment this if you really want to proceed.')
  
    
    #if (nargin < 5), verbose = 1; end
    # Optional argument
    
    #implicit = isa(A,'function_handle');
    if hasattr(A, '__call__'):
        implicit = True
    else:
        implicit = False
    
    x = np.zeros(b.size)
    r = b.copy()
    d = r.copy()
    delta = np.vdot(r,r)
    delta0 = np.vdot(b,b)
    numiter = 0
    bestx = x.copy()
    bestres = math.sqrt(delta/delta0)
    while (numiter < maxiter) and (delta > tol**2*delta0):
    
      # q = A*d
      #if (implicit), q = A(d);  else  q = A*d;  end
      if implicit:
          q = A(d)
      else:
          q = np.dot(A,d)
     
      alpha = delta/np.vdot(d,q)
      x = x + alpha*d
      
      if divmod(numiter+1,50)[1] == 0:
        # r = b - Aux*x
        #if (implicit), r = b - A(x);  else  r = b - A*x;  end
        if implicit:
            r = b - A(x)
        else:
            r = b - np.dot(A,x)
      else:
        r = r - alpha*q
      #end
      
      deltaold = delta;
      delta = np.vdot(r,r)
      beta = delta/deltaold;
      d = r + beta*d
      numiter = numiter + 1
      if (math.sqrt(delta/delta0) < bestres):
        bestx = x.copy()
        bestres = math.sqrt(delta/delta0)
      #end    
      
      if ((verbose) and (divmod(numiter,verbose)[1]==0)):
        #disp(sprintf('cg: Iter = #d, Best residual = #8.3e, Current residual = #8.3e', ...
        #  numiter, bestres, sqrt(delta/delta0)));
        print 'cg: Iter = ',numiter,', Best residual = ',bestres,', Current residual = ',math.sqrt(delta/delta0)
      #end
      
    #end
    
    if (verbose):
      #disp(sprintf('cg: Iterations = #d, best residual = #14.8e', numiter, bestres));
      print 'cg: Iterations = ',numiter,', best residual = ',bestres
    #end
    x = bestx.copy()
    res = bestres
    iter = numiter
    
    return x,res,iter



def l1qec_newton(x0, u0, A, At, b, epsilon, Aexact, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose=False):

    # check if the matrix A is implicit or explicit
    #largescale = isa(A,'function_handle');
    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False    
    
    # line search parameters
    alpha = 0.01
    beta = 0.5
    
    #if (~largescale), AtA = A'*A; end
    if not largescale:
        AtA = np.dot(A.T,A)
    
    # initial point
    x = x0.copy()
    u = u0.copy()
    #if (largescale), r = A(x) - b; else  r = A*x - b; end
    if largescale:
        r = A(x) - b
    else:
        r = np.dot(A,x) - b
        
    fu1 = x - u
    fu2 = -x - u
    fe = 1.0/2*(np.vdot(r,r) - epsilon**2)
    f = u.sum() - (1.0/tau)*(np.log(-fu1).sum() + np.log(-fu2).sum() + math.log(-fe))
    
    niter = 0
    done = 0
    while not done:
      
      #if (largescale), atr = At(r); else  atr = A'*r; end
      if largescale:
          atr = At(r)
      else:
          atr = np.dot(A.T,r)
      
      #ntgz = 1./fu1 - 1./fu2 + 1/fe*atr;
      ntgz = 1.0/fu1 - 1.0/fu2 + 1.0/fe*atr
      #ntgu = -tau - 1./fu1 - 1./fu2;
      ntgu = -tau - 1.0/fu1 - 1.0/fu2
      #gradf = -(1/tau)*[ntgz; ntgu];
      gradf = -(1.0/tau)*np.concatenate((ntgz, ntgu),0)
      
      #sig11 = 1./fu1.^2 + 1./fu2.^2;
      sig11 = 1.0/(fu1**2) + 1.0/(fu2**2)
      #sig12 = -1./fu1.^2 + 1./fu2.^2;
      sig12 = -1.0/(fu1**2) + 1.0/(fu2**2)
      #sigx = sig11 - sig12.^2./sig11;
      sigx = sig11 - (sig12**2)/sig11
        
      #w1p = ntgz - sig12./sig11.*ntgu;
      w1p = ntgz - sig12/sig11*ntgu
      if largescale:
        #h11pfun = @(z) sigx.*z - (1/fe)*At(A(z)) + 1/fe^2*(atr'*z)*atr;
        h11pfun = lambda z: sigx*z - (1.0/fe)*At(A(z)) + 1.0/(fe**2)*np.dot(np.dot(atr.T,z),atr)
        dx,cgres,cgiter = cgsolve(h11pfun, w1p, cgtol, cgmaxiter, 0)
        if (cgres > 1.0/2):
          if verbose:
            print 'Cannot solve system.  Returning previous iterate.  (See Section 4 of notes for more information.)'
          xp = x.copy()
          up = u.copy()
          return xp,up,niter
        #end
        Adx = A(dx)
      else:
        #H11p = diag(sigx) - (1/fe)*AtA + (1/fe)^2*atr*atr';
        # Attention: atr is column vector, so atr*atr' means outer(atr,atr)
        H11p = np.diag(sigx) - (1.0/fe)*AtA + (1.0/fe)**2*np.outer(atr,atr)
        #opts.POSDEF = true; opts.SYM = true;
        #[dx,hcond] = linsolve(H11p, w1p, opts);
        #if (hcond < 1e-14)
        #  disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
        #  xp = x;  up = u;
        #  return
        #end
        
        # Nic says: from tveq_newton.m
        K = Aexact.shape[0]
        afac = (np.diag(H11p)).max()
        #Hp = [H11p afac*A'; afac*A zeros(K)])
        Hp = np.vstack(( np.hstack((H11p,afac*Aexact.T)) , np.hstack((afac*Aexact,np.zeros((K,K)))) ))
        wp = np.concatenate((w1p , np.zeros(K)))
        try:
            #dx = scipy.linalg.solve(H11p, w1p, sym_pos=True)
            #hcond = 1.0/np.linalg.cond(H11p)
            dxv = scipy.linalg.solve(Hp, wp, sym_pos=False) # Only symmetric, not posdef
            dx = dxv[:x0.size]
            hcond = 1.0/np.linalg.cond(Hp)
        except scipy.linalg.LinAlgError:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            up = u.copy()
            return xp,up,niter
        if hcond < 1e-14:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            up = u.copy()
            return xp,up,niter
        
        #Adx = A*dx;
        Adx = np.dot(A,dx)
      #end
      #du = (1./sig11).*ntgu - (sig12./sig11).*dx;  
      du = (1.0/sig11)*ntgu - (sig12/sig11)*dx;
     
      # minimum step size that stays in the interior
      #ifu1 = find((dx-du) > 0); ifu2 = find((-dx-du) > 0);
      ifu1 = np.nonzero((dx-du)>0)
      ifu2 = np.nonzero((-dx-du)>0)
      #aqe = Adx'*Adx;   bqe = 2*r'*Adx;   cqe = r'*r - epsilon^2;
      aqe = np.dot(Adx.T,Adx)
      bqe = 2*np.dot(r.T,Adx)
      cqe = np.vdot(r,r) - epsilon**2
      #smax = min(1,min([...
      #  -fu1(ifu1)./(dx(ifu1)-du(ifu1)); -fu2(ifu2)./(-dx(ifu2)-du(ifu2)); ...
      #  (-bqe+sqrt(bqe^2-4*aqe*cqe))/(2*aqe)
      #  ]));
      smax = min(1,np.concatenate( (-fu1[ifu1]/(dx[ifu1]-du[ifu1]) , -fu2[ifu2]/(-dx[ifu2]-du[ifu2]) , np.array([ (-bqe + math.sqrt(bqe**2-4*aqe*cqe))/(2*aqe) ]) ) , 0).min())
      
      s = 0.99 * smax
      
      # backtracking line search
      suffdec = 0
      backiter = 0
      while not suffdec:
        #xp = x + s*dx;  up = u + s*du;  rp = r + s*Adx;
        xp = x + s*dx
        up = u + s*du
        rp = r + s*Adx
        #fu1p = xp - up;  fu2p = -xp - up;  fep = 1/2*(rp'*rp - epsilon^2);
        fu1p = xp - up
        fu2p = -xp - up
        fep = 0.5*(np.vdot(rp,rp) - epsilon**2)
        #fp = sum(up) - (1/tau)*(sum(log(-fu1p)) + sum(log(-fu2p)) + log(-fep));
        fp = up.sum() - (1.0/tau)*(np.log(-fu1p).sum() + np.log(-fu2p).sum() + math.log(-fep))
        #flin = f + alpha*s*(gradf'*[dx; du]);
        flin = f + alpha*s*np.dot(gradf.T , np.concatenate((dx,du),0))
        #suffdec = (fp <= flin);
        if fp <= flin:
            suffdec = True
        else:
            suffdec = False
        
        s = beta*s
        backiter = backiter + 1
        if (backiter > 32):
          if verbose:
            print 'Stuck on backtracking line search, returning previous iterate.  (See Section 4 of notes for more information.)'
          xp = x.copy()
          up = u.copy()
          return xp,up,niter
        #end
      #end
      
      # set up for next iteration
      #x = xp; u = up;  r = rp;
      x = xp.copy()
      u = up.copy()
      r = rp.copy()
      #fu1 = fu1p;  fu2 = fu2p;  fe = fep;  f = fp;
      fu1 = fu1p.copy()
      fu2 = fu2p.copy()
      fe = fep
      f = fp
      
      #lambda2 = -(gradf'*[dx; du]);
      lambda2 = -np.dot(gradf.T , np.concatenate((dx,du),0))
      #stepsize = s*norm([dx; du]);
      stepsize = s * np.linalg.norm(np.concatenate((dx,du),0))
      niter = niter + 1
      #done = (lambda2/2 < newtontol) | (niter >= newtonmaxiter);
      if lambda2/2.0 < newtontol or niter >= newtonmaxiter:
          done = 1
      else:
          done = 0
      
      #disp(sprintf('Newton iter = #d, Functional = #8.3f, Newton decrement = #8.3f, Stepsize = #8.3e', ...
      if verbose:
        print 'Newton iter = ',niter,', Functional = ',f,', Newton decrement = ',lambda2/2.0,', Stepsize = ',stepsize

      if verbose:
        if largescale:
            print '                CG Res = ',cgres,', CG Iter = ',cgiter
        else:
            print '                  H11p condition number = ',hcond
      #end
          
    #end
    return xp,up,niter

def l1qec_logbarrier(x0, A, At, b, epsilon, Aexact, Atexact, bexact, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=False):
    
    # Check if epsilon > 0. If epsilon is 0, the algorithm fails. You should run the algo with equality constraint instead  
    if epsilon == 0:
      raise l1qecInputValueError('Epsilon should be > 0!')      
    
    #largescale = isa(A,'function_handle');
    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False
    
    #    if (nargin < 6), lbtol = 1e-3; end
    #    if (nargin < 7), mu = 10; end
    #    if (nargin < 8), cgtol = 1e-8; end
    #    if (nargin < 9), cgmaxiter = 200; end
    # Nic: added them as optional parameteres
    
    newtontol = lbtol
    newtonmaxiter = 50
    
    #N = length(x0);
    N = x0.size
    
    # starting point --- make sure that it is feasible
    if largescale:
      if np.linalg.norm(A(x0) - b) > epsilon or np.linalg.norm( np.dot(Aexact,x0) - bexact ) > 1e-15:
        if verbose:
          print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
        #AAt = @(z) A(At(z));
        AAt = lambda z: A(At(z))
        # TODO: implement cgsolve
        w,cgres,cgiter = cgsolve(AAt, b, cgtol, cgmaxiter, 0)
        if (cgres > 1.0/2):
          if verbose:
            print 'A*At is ill-conditioned: cannot find starting point'
          xp = x0.copy()
          return xp
        #end
        x0 = At(w)
      #end
    else:
      # Nic: add test for np.dot(Aexact,x0) - bexact ) > 1e-15
      if np.linalg.norm( np.dot(A,x0) - b ) > epsilon or np.linalg.norm( np.dot(Aexact,x0) - bexact ) > 1e-15:
        if verbose:
          print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
        
        #Nic: stack A and Aexact, b and bexact, and use them instead of A and b
        Abig = np.vstack((A,Aexact))
        bbig = np.concatenate((b,bexact))
        try:
            w = scipy.linalg.solve(np.dot(Abig,Abig.T), bbig, sym_pos=True)
            #w = np.linalg.solve(np.dot(A,A.T), b)
            hcond = 1.0/np.linalg.cond(np.dot(Abig,Abig.T))
        except scipy.linalg.LinAlgError:
            if verbose:
              print 'A*At is ill-conditioned: cannot find starting point'
            xp = x0.copy()
            return xp
        if hcond < 1e-14:
            if verbose:
              print 'A*At is ill-conditioned: cannot find starting point'
            xp = x0.copy()
            return xp           
        x0 = np.dot(Abig.T, w)        
        #        try:
        #            w = scipy.linalg.solve(np.dot(A,A.T), b, sym_pos=True)
        #            #w = np.linalg.solve(np.dot(A,A.T), b)
        #            hcond = 1.0/scipy.linalg.cond(np.dot(A,A.T))
        #        except scipy.linalg.LinAlgError:
        #            print 'A*At is ill-conditioned: cannot find starting point'
        #            xp = x0.copy()
        #            return xp
        #        if hcond < 1e-14:
        #            print 'A*At is ill-conditioned: cannot find starting point'
        #            xp = x0.copy()
        #            return xp           
        #        #x0 = A'*w;
        #        x0 = np.dot(A.T, w)
      #end  
    #end
    x = x0.copy()
    u = (0.95)*np.abs(x0) + (0.10)*np.abs(x0).max()
    
    #disp(sprintf('Original l1 norm = #.3f, original functional = #.3f', sum(abs(x0)), sum(u)));
    if verbose:
      print 'Original l1 norm = ',np.abs(x0).sum(),'original functional = ',u.sum()
    
    # choose initial value of tau so that the duality gap after the first
    # step will be about the origial norm
    tau = max(((2*N+1.0)/np.abs(x0).sum()), 1)
                                                                                                                              
    lbiter = math.ceil((math.log(2*N+1)-math.log(lbtol)-math.log(tau))/math.log(mu))
    #disp(sprintf('Number of log barrier iterations = #d\n', lbiter));
    if verbose:
      print 'Number of log barrier iterations = ',lbiter
    
    totaliter = 0
    
    # Added by Nic, to fix some crashing
    if lbiter == 0:
        xp = np.zeros(x0.size)
    #end
    
    #for ii = 1:lbiter
    for ii in np.arange(lbiter):
    
      xp,up,ntiter = l1qec_newton(x, u, A, At, b, epsilon, Aexact, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
      totaliter = totaliter + ntiter
      
      #disp(sprintf('\nLog barrier iter = #d, l1 = #.3f, functional = #8.3f, tau = #8.3e, total newton iter = #d\n', ...
      #  ii, sum(abs(xp)), sum(up), tau, totaliter));
      if verbose:
        print 'Log barrier iter = ',ii,', l1 = ',np.abs(xp).sum(),', functional = ',up.sum(),', tau = ',tau,', total newton iter = ',totaliter
      x = xp.copy()
      u = up.copy()
     
      tau = mu*tau
      
    #end
    return xp
                   
