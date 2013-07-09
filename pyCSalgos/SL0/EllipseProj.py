# -*- coding: utf-8 -*-
"""

Author: Nicolae Cleju
"""
__author__ = "Nicolae Cleju"
__license__ = "GPL"
__email__ = "nikcleju@gmail.com"


import numpy
import scipy
import math
#import cvxpy

class EllipseProjDaiError(Exception):
#  def __init__(self, e):
#    self._e = e
  pass

def ellipse_proj_cvxpy(A,b,p,epsilon):
  b = b.reshape((b.size,1))
  p = p.reshape((p.size,1))
  A = cvxpy.matrix(A)
  b = cvxpy.matrix(b)
  p = cvxpy.matrix(p)

  x = cvxpy.variable(p.shape[0],1)                 # Create optimization variable
  prog = cvxpy.program(cvxpy.minimize(cvxpy.norm2(p - x)),              # Create problem instance
            [cvxpy.leq(cvxpy.norm2(b-A*x),epsilon)])
  #prog.solve(quiet=True)                              # Get optimal value
  prog.solve()                              # Get optimal value
  return numpy.array(x.value).flatten()

def ellipse_proj_dai(A,x,s,eps):
  
  A_pinv = numpy.linalg.pinv(A)
  
  AA = numpy.dot(A.T, A)
  bb = -numpy.dot(x.T,A)
  alpha = eps*eps - numpy.inner(x,x)
  
  direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
  s0 = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction
  
  a = s
  xk = s0
  m1 = 1
  m2 = 1
  c1 = 0.1
  c2 = 0.8
  done = False
  k = 0
  while not done and k < 50:
    
    uk = numpy.dot(AA,xk) + bb
    vk = a - xk
    
    zk = uk - (numpy.inner(uk,vk)/numpy.inner(vk,vk))*vk
    vtav = numpy.inner(vk,numpy.dot(AA,vk))
    vtv = numpy.inner(vk,vk)
    ztaz = numpy.inner(zk,numpy.dot(AA,zk))
    ztz = numpy.inner(zk,zk)
    vtaz = numpy.inner(vk,numpy.dot(AA,zk))
    gammak1 = 1.0/(0.5 * ( vtav/vtv + ztaz/ztz + math.sqrt((vtaz/vtv - ztaz/ztz)**2 + 4*vtaz**2/vtv/ztz)))
    
    pk = vk / numpy.linalg.norm(vk)
    qk = zk / numpy.linalg.norm(zk)
    Hk = numpy.zeros((pk.size,2))
    Hk[:,0] = pk
    Hk[:,1] = qk
    Ak = numpy.dot(Hk.T, numpy.dot(AA,Hk))
    bk = numpy.dot(Hk.T,uk)
    
    #al = numpy.dot(Hk, numpy.dot(Hk.T, a))
    al = numpy.array([numpy.linalg.norm(vk), 0])
    D,Q = numpy.linalg.eig(Ak)
    Q = Q.T
    ah = numpy.dot(Q,al) + (1.0/D) * numpy.dot(Q,bk)
    
    l1 = D[0]
    l2 = D[1]
    Qbk = numpy.dot(Q,bk)
    beta = numpy.dot(Qbk.T, (1.0/D) * Qbk)
    hstar1s = numpy.roots(numpy.array([ (l1-l2)**2*l1, 
                                  2*(l1-l2)*l2*ah[0]*l1,
                                  -(l1-l2)**2*beta + l2**2*ah[0]**2*l1 + l1**2*l2*ah[1]**2,
                                  -2*beta*(l1-l2)*l2*ah[0],
                                  -beta*l2**2*ah[0]**2]))
    hstar2s = numpy.zeros_like(hstar1s)
    i_s = []
    illcond = False  # flag if ill conditioned problem (numerical errros)
    for i in range(hstar1s.size):
      
      # Protect against bad conditioning (ratio of two very small values)
      if numpy.abs(l1*ah[1]) > 1e-6 and numpy.abs((l1-l2) + l2*ah[0]/hstar1s[i]) > 1e-6:
        
        # Ignore small imaginary parts
        if numpy.abs(numpy.imag(hstar1s[i])) < 1e-10:
          hstar1s[i] = numpy.real(hstar1s[i])
        
        hstar2s[i] = l1*ah[1] / ((l1-l2)*hstar1s[i] + l2*ah[0]) * hstar1s[i]
        
        # Ignore small imaginary parts
        if numpy.abs(numpy.imag(hstar2s[i])) < 1e-10:
          hstar2s[i] = numpy.real(hstar2s[i])
          
        if (ah[0] - hstar1s[i]) / (l1*hstar1s[i]) > 0 and (ah[1] - hstar2s[i]) / (l2*hstar2s[i]) > 0:
          i_s.append(i)
        
      else:
        # Cannot rely on hstar2s[i] calculation since it is the ratio of two small numbers
        # Do a vertical projection instead
          hstar1 = ah[0]
          hstar2 = numpy.sign(ah[1]) * math.sqrt((beta - l1*ah[0]**2)/l2)
          illcond = True  # Flag, so we don't take i_s[] into account anymore
    
    if illcond:
      print "Ill conditioned problem, do vertical projection instead"
      # hstar1 and hstar2 are already set above, nothing to do here
    else:
      if len(i_s) > 1:
        hstar1 = hstar1s[i_s[0]].real
        hstar2 = hstar2s[i_s[0]].real
      elif len(i_s) == 0:
        # Again do vertical projection
        hstar1 = ah[0]
        hstar2 = numpy.sign(ah[1]) * math.sqrt((beta - l1*ah[0]**2)/l2)
      else:
        # Everything is ok
        hstar1 = hstar1s[i_s].real
        hstar2 = hstar2s[i_s].real
      
    ahstar = numpy.array([hstar1, hstar2]).flatten()
    alstar = numpy.dot(Q.T, ahstar - (1.0/D) * numpy.dot(Q,bk))
    alstar1 = alstar[0]
    alstar2 = alstar[1]
    etak = 1 - alstar1/numpy.linalg.norm(vk) + numpy.inner(uk,vk)/vtv*alstar2/numpy.linalg.norm(zk)
    gammak2 = -alstar2/numpy.linalg.norm(zk)/etak
    if k % (m1+m2) < m1:
      gammak = gammak2
    else:
      gammak = c1*gammak1 + c2*gammak2
      
    # Safeguard:
    if gammak < 0:
      gammak = gammak1

    ck = xk - gammak * uk
    wk = ck - a
    
    ga = numpy.dot(AA,a) + bb
    qa = numpy.inner(a,numpy.dot(AA,a)) + 2*numpy.inner(bb,a)
    # Check if delta < 0 but very small (possibly numerical errors)
    if (numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk)))**2 - (qa-alpha)/numpy.inner(wk, numpy.dot(AA,wk)) < 0 and abs((numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk)))**2 - (qa-alpha)/numpy.inner(wk, numpy.dot(AA,wk))) < 1e-10:
      etak = -numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk))
    else:
      assert ((numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk)))**2 - (qa-alpha)/numpy.inner(wk, numpy.dot(AA,wk)) >= 0)
      etak = -numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk)) - math.sqrt((numpy.inner(ga,wk)/numpy.inner(wk, numpy.dot(AA,wk)))**2 - (qa-alpha)/numpy.inner(wk, numpy.dot(AA,wk)))
    xk = a + etak * wk
    
    if (1 - numpy.inner(uk,vk) / numpy.linalg.norm(uk) / numpy.linalg.norm(vk)) <= 0.01:
      done = True
    k = k+1
  
  return xk

  
def ellipse_proj_proj(A,x,s,eps,L2):
  A_pinv = numpy.linalg.pinv(A)
  u,singvals,v  = numpy.linalg.svd(A, full_matrices=0)
  singvals = numpy.flipud(singvals)
  s_orig = s
  
  for j in range(L2):
    direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))

    if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
      
      P = numpy.dot(numpy.dot(u,v), s)
      sproj = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction
      P0 = numpy.dot(numpy.dot(u,v), sproj)
      
      tangent = (P0 * (singvals**2)).reshape((1,P.size))
      uu,ss,vv = numpy.linalg.svd(tangent)
      svd = vv[1:,:]
      P1 = numpy.linalg.solve(numpy.vstack((tangent,svd)), numpy.vstack((numpy.array([[eps]]), numpy.dot(svd, P).reshape((svd.shape[0],1)))))
      
      # Take only a smaller step
      #P1 = P0.reshape((P0.size,1)) + 0.1*(P1-P0.reshape((P0.size,1)))
      
      #s = numpy.dot(A_pinv,P1).flatten() + numpy.dot(A_pinv,numpy.dot(A,s)).flatten()
      s = numpy.dot(numpy.linalg.pinv(numpy.dot(u,v)),P1).flatten() + (s-numpy.dot(A_pinv,numpy.dot(A,s)).flatten())
      
      #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
  
  direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
  if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
    s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction  
    
  return s

def ellipse_proj_logbarrier(A,b,p,epsilon,verbose=False):
  return l1qc_logbarrier(numpy.zeros_like(p), p, A, A.T, b, epsilon, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=verbose)

def l1qc_newton(x0, p, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose=False):
    # Newton algorithm for log-barrier subproblems for l1 minimization
    # with quadratic constraints.
    #
    # Usage: 
    # [xp,up,niter] = l1qc_newton(x0, u0, A, At, b, epsilon, tau, 
    #                             newtontol, newtonmaxiter, cgtol, cgmaxiter)
    #
    # x0,u0 - starting points
    #
    # A - Either a handle to a function that takes a N vector and returns a K 
    #     vector , or a KxN matrix.  If A is a function handle, the algorithm
    #     operates in "largescale" mode, solving the Newton systems via the
    #     Conjugate Gradients algorithm.
    #
    # At - Handle to a function that takes a K vector and returns an N vector.
    #      If A is a KxN matrix, At is ignored.
    #
    # b - Kx1 vector of observations.
    #
    # epsilon - scalar, constraint relaxation parameter
    #
    # tau - Log barrier parameter.
    #
    # newtontol - Terminate when the Newton decrement is <= newtontol.
    #         Default = 1e-3.
    #
    # newtonmaxiter - Maximum number of iterations.
    #         Default = 50.
    #
    # cgtol - Tolerance for Conjugate Gradients; ignored if A is a matrix.
    #     Default = 1e-8.
    #
    # cgmaxiter - Maximum number of iterations for Conjugate Gradients; ignored
    #     if A is a matrix.
    #     Default = 200.
    #
    # Written by: Justin Romberg, Caltech
    # Email: jrom@acm.caltech.edu
    # Created: October 2005
    #
    
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
        AtA = numpy.dot(A.T,A)
    
    # initial point
    x = x0.copy()
    #u = u0.copy()
    #if (largescale), r = A(x) - b; else  r = A*x - b; end
    if largescale:
        r = A(x) - b
    else:
        r = numpy.dot(A,x) - b
        
    #fu1 = x - u
    #fu2 = -x - u
    fe = 1.0/2*(numpy.vdot(r,r) - epsilon**2)
    #f = u.sum() - (1.0/tau)*(numpy.log(-fu1).sum() + numpy.log(-fu2).sum() + math.log(-fe))
    f = numpy.linalg.norm(p-x)**2 - (1.0/tau) * math.log(-fe)
    
    niter = 0
    done = 0
    while not done:
      
      #if (largescale), atr = At(r); else  atr = A'*r; end
      if largescale:
          atr = At(r)
      else:
          atr = numpy.dot(A.T,r)
      
      ##ntgz = 1./fu1 - 1./fu2 + 1/fe*atr;
      #ntgz = 1.0/fu1 - 1.0/fu2 + 1.0/fe*atr
      ntgz = 1.0/fe*atr
      ##ntgu = -tau - 1./fu1 - 1./fu2;
      #ntgu = -tau - 1.0/fu1 - 1.0/fu2
      ##gradf = -(1/tau)*[ntgz; ntgu];
      #gradf = -(1.0/tau)*numpy.concatenate((ntgz, ntgu),0)
      gradf = 2*x + 2*p -(1.0/tau)*ntgz
      
      ##sig11 = 1./fu1.^2 + 1./fu2.^2;
      #sig11 = 1.0/(fu1**2) + 1.0/(fu2**2)
      ##sig12 = -1./fu1.^2 + 1./fu2.^2;
      #sig12 = -1.0/(fu1**2) + 1.0/(fu2**2)
      ##sigx = sig11 - sig12.^2./sig11;
      #sigx = sig11 - (sig12**2)/sig11
        
      ##w1p = ntgz - sig12./sig11.*ntgu;
      #w1p = ntgz - sig12/sig11*ntgu
      #w1p = -tau * (2*x + 2*p) + ntgz
      w1p = -gradf
      if largescale:
        #h11pfun = @(z) sigx.*z - (1/fe)*At(A(z)) + 1/fe^2*(atr'*z)*atr;
        h11pfun = lambda z: sigx*z - (1.0/fe)*At(A(z)) + 1.0/(fe**2)*numpy.dot(numpy.dot(atr.T,z),atr)
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
        ##H11p = diag(sigx) - (1/fe)*AtA + (1/fe)^2*atr*atr';
        # Attention: atr is column vector, so atr*atr' means outer(atr,atr)
        #H11p = numpy.diag(sigx) - (1.0/fe)*AtA + (1.0/fe)**2*numpy.outer(atr,atr)
        H11p = 2 * numpy.eye(x.size) - (1.0/fe)*AtA + (1.0/fe)**2*numpy.outer(atr,atr)
        #opts.POSDEF = true; opts.SYM = true;
        #[dx,hcond] = linsolve(H11p, w1p, opts);
        #if (hcond < 1e-14)
        #  disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
        #  xp = x;  up = u;
        #  return
        #end
        try:
            dx = scipy.linalg.solve(H11p, w1p, sym_pos=True)
            #dx = numpy.linalg.solve(H11p, w1p)
            hcond = 1.0/numpy.linalg.cond(H11p)
        except scipy.linalg.LinAlgError:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            #up = u.copy()
            #return xp,up,niter
            return xp,niter
        if hcond < 1e-14:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            #up = u.copy()
            #return xp,up,niter
            return xp,niter
        
        #Adx = A*dx;
        Adx = numpy.dot(A,dx)
      #end
      ##du = (1./sig11).*ntgu - (sig12./sig11).*dx;  
      #du = (1.0/sig11)*ntgu - (sig12/sig11)*dx;
     
      # minimum step size that stays in the interior
      ##ifu1 = find((dx-du) > 0); ifu2 = find((-dx-du) > 0);
      #ifu1 = numpy.nonzero((dx-du)>0)
      #ifu2 = numpy.nonzero((-dx-du)>0)
      #aqe = Adx'*Adx;   bqe = 2*r'*Adx;   cqe = r'*r - epsilon^2;
      aqe = numpy.dot(Adx.T,Adx)
      bqe = 2*numpy.dot(r.T,Adx)
      cqe = numpy.vdot(r,r) - epsilon**2
      #smax = min(1,min([...
      #  -fu1(ifu1)./(dx(ifu1)-du(ifu1)); -fu2(ifu2)./(-dx(ifu2)-du(ifu2)); ...
      #  (-bqe+sqrt(bqe^2-4*aqe*cqe))/(2*aqe)
      #  ]));
      #smax = min(1,numpy.concatenate( (-fu1[ifu1]/(dx[ifu1]-du[ifu1]) , -fu2[ifu2]/(-dx[ifu2]-du[ifu2]) , numpy.array([ (-bqe + math.sqrt(bqe**2-4*aqe*cqe))/(2*aqe) ]) ) , 0).min())
      smax = min(1,numpy.array([ (-bqe + math.sqrt(bqe**2-4*aqe*cqe))/(2*aqe) ] ).min())
      
      s = 0.99 * smax
      
      # backtracking line search
      suffdec = 0
      backiter = 0
      while not suffdec:
        #xp = x + s*dx;  up = u + s*du;  rp = r + s*Adx;
        xp = x + s*dx
        #up = u + s*du
        rp = r + s*Adx
        #fu1p = xp - up;  fu2p = -xp - up;  fep = 1/2*(rp'*rp - epsilon^2);
        #fu1p = xp - up
        #fu2p = -xp - up
        fep = 0.5*(numpy.vdot(rp,rp) - epsilon**2)
        ##fp = sum(up) - (1/tau)*(sum(log(-fu1p)) + sum(log(-fu2p)) + log(-fep));
        #fp = up.sum() - (1.0/tau)*(numpy.log(-fu1p).sum() + numpy.log(-fu2p).sum() + math.log(-fep))
        fp = numpy.linalg.norm(p-xp)**2 - (1.0/tau) * math.log(-fep)
        #flin = f + alpha*s*(gradf'*[dx; du]);
        flin = f + alpha*s*numpy.dot(gradf.T , dx)
        #suffdec = (fp <= flin);
        if fp <= flin:
            suffdec = True
        else:
            suffdec = False
        
        s = beta*s
        backiter = backiter + 1
        if (backiter > 132):
          if verbose:
            print 'Stuck on backtracking line search, returning previous iterate.  (See Section 4 of notes for more information.)'
          xp = x.copy()
          #up = u.copy()
          #return xp,up,niter
          return xp,niter
        #end
      #end
      
      # set up for next iteration
      ##x = xp; u = up;  r = rp;
      x = xp.copy()
      #u = up.copy()
      r = rp.copy()
      ##fu1 = fu1p;  fu2 = fu2p;  fe = fep;  f = fp;
      #fu1 = fu1p.copy()
      #fu2 = fu2p.copy()
      fe = fep
      f = fp
      
      ##lambda2 = -(gradf'*[dx; du]);
      #lambda2 = -numpy.dot(gradf.T , numpy.concatenate((dx,du),0))
      lambda2 = -numpy.dot(gradf.T , dx)
      ##stepsize = s*norm([dx; du]);
      #stepsize = s * numpy.linalg.norm(numpy.concatenate((dx,du),0))
      stepsize = s * numpy.linalg.norm(dx)
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
    #return xp,up,niter
    return xp,niter

#function xp = l1qc_logbarrier(x0, A, At, b, epsilon, lbtol, mu, cgtol, cgmaxiter)
def l1qc_logbarrier(x0, p, A, At, b, epsilon, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=False):
    # Solve quadratically constrained l1 minimization:
    # min ||x||_1   s.t.  ||Ax - b||_2 <= \epsilon
    #
    # Reformulate as the second-order cone program
    # min_{x,u}  sum(u)   s.t.    x - u <= 0,
    #                            -x - u <= 0,
    #      1/2(||Ax-b||^2 - \epsilon^2) <= 0
    # and use a log barrier algorithm.
    #
    # Usage:  xp = l1qc_logbarrier(x0, A, At, b, epsilon, lbtol, mu, cgtol, cgmaxiter)
    #
    # x0 - Nx1 vector, initial point.
    #
    # A - Either a handle to a function that takes a N vector and returns a K 
    #     vector , or a KxN matrix.  If A is a function handle, the algorithm
    #     operates in "largescale" mode, solving the Newton systems via the
    #     Conjugate Gradients algorithm.
    #
    # At - Handle to a function that takes a K vector and returns an N vector.
    #      If A is a KxN matrix, At is ignored.
    #
    # b - Kx1 vector of observations.
    #
    # epsilon - scalar, constraint relaxation parameter
    #
    # lbtol - The log barrier algorithm terminates when the duality gap <= lbtol.
    #         Also, the number of log barrier iterations is completely
    #         determined by lbtol.
    #         Default = 1e-3.
    #
    # mu - Factor by which to increase the barrier constant at each iteration.
    #      Default = 10.
    #
    # cgtol - Tolerance for Conjugate Gradients; ignored if A is a matrix.
    #     Default = 1e-8.
    #
    # cgmaxiter - Maximum number of iterations for Conjugate Gradients; ignored
    #     if A is a matrix.
    #     Default = 200.
    #
    # Written by: Justin Romberg, Caltech
    # Email: jrom@acm.caltech.edu
    # Created: October 2005
    #

    
    # Check if epsilon > 0. If epsilon is 0, the algorithm fails. You should run the algo with equality constraint instead  
    if epsilon == 0:
      raise l1qcInputValueError('Epsilon should be > 0!')       
    
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
    newtonmaxiter = 150
    
    #N = length(x0);
    N = x0.size
    
    # starting point --- make sure that it is feasible
    if largescale:
      if numpy.linalg.norm(A(x0) - b) > epsilon:
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
      if numpy.linalg.norm( numpy.dot(A,x0) - b ) > epsilon:
        if verbose:
          print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
        #opts.POSDEF = true; opts.SYM = true;
        #[w, hcond] = linsolve(A*A', b, opts);
        #if (hcond < 1e-14)
        #  disp('A*At is ill-conditioned: cannot find starting point');
        #  xp = x0;
        #  return;
        #end
        try:
            w = scipy.linalg.solve(numpy.dot(A,A.T), b, sym_pos=True)
            #w = numpy.linalg.solve(numpy.dot(A,A.T), b)
            hcond = 1.0/numpy.linalg.cond(numpy.dot(A,A.T))
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
        #x0 = A'*w;
        x0 = numpy.dot(A.T, w)
      #end  
    #end
    x = x0.copy()
    #u = (0.95)*numpy.abs(x0) + (0.10)*numpy.abs(x0).max()
    
    #disp(sprintf('Original l1 norm = #.3f, original functional = #.3f', sum(abs(x0)), sum(u)));
    if verbose:
      #print 'Original l1 norm = ',numpy.abs(x0).sum(),'original functional = ',u.sum()
      print 'Original distance ||p-x|| = ',numpy.linalg.norm(p-x)
    # choose initial value of tau so that the duality gap after the first
    # step will be about the origial norm
    #tau = max(((2*N+1.0)/numpy.abs(x0).sum()), 1)
    # Nic:
    tau = max(((2*N+1.0)/numpy.linalg.norm(p-x0)**2), 1)
                                                                                                                              
    lbiter = math.ceil((math.log(2*N+1)-math.log(lbtol)-math.log(tau))/math.log(mu))
    #disp(sprintf('Number of log barrier iterations = #d\n', lbiter));
    if verbose:
      print 'Number of log barrier iterations = ',lbiter
    
    totaliter = 0
    
    # Added by Nic, to fix some crashing
    if lbiter == 0:
        xp = numpy.zeros(x0.size)
    #end
    
    #for ii = 1:lbiter
    for ii in numpy.arange(lbiter):
    
      #xp,up,ntiter = l1qc_newton(x, u, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
      xp,ntiter = l1qc_newton(x, p, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose=verbose)
      totaliter = totaliter + ntiter
      
      #disp(sprintf('\nLog barrier iter = #d, l1 = #.3f, functional = #8.3f, tau = #8.3e, total newton iter = #d\n', ...
      #  ii, sum(abs(xp)), sum(up), tau, totaliter));
      if verbose:
        #print 'Log barrier iter = ',ii,', l1 = ',numpy.abs(xp).sum(),', functional = ',up.sum(),', tau = ',tau,', total newton iter = ',totaliter
        print 'Log barrier iter = ',ii,', l1 = ',numpy.abs(xp).sum(),', functional = ',numpy.linalg.norm(p-xp),', tau = ',tau,', total newton iter = ',totaliter
      x = xp.copy()
      #u = up.copy()
     
      tau = mu*tau
      
    #end
    return xp  