
import numpy
import scipy.linalg
import math

class l1eqNotImplementedError(Exception):
  pass

#function xp = l1eq_pd(x0, A, At, b, pdtol, pdmaxiter, cgtol, cgmaxiter)
def l1eq_pd(x0, A, At, b, pdtol=1e-3, pdmaxiter=50, cgtol=1e-8, cgmaxiter=200, verbose=False):

    # Solve
    # min_x ||x||_1  s.t.  Ax = b
    #
    # Recast as linear program
    # min_{x,u} sum(u)  s.t.  -u <= x <= u,  Ax=b
    # and use primal-dual interior point method
    #
    # Usage: xp = l1eq_pd(x0, A, At, b, pdtol, pdmaxiter, cgtol, cgmaxiter)
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
    # pdtol - Tolerance for primal-dual algorithm (algorithm terminates if
    #     the duality gap is less than pdtol).  
    #     Default = 1e-3.
    #
    # pdmaxiter - Maximum number of primal-dual iterations.  
    #     Default = 50.
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
    
    
    #---------------------
    # Original Matab code:    
        
    #largescale = isa(A,'function_handle');
    #
    #if (nargin < 5), pdtol = 1e-3;  end
    #if (nargin < 6), pdmaxiter = 50;  end
    #if (nargin < 7), cgtol = 1e-8;  end
    #if (nargin < 8), cgmaxiter = 200;  end
    #
    #N = length(x0);
    #
    #alpha = 0.01;
    #beta = 0.5;
    #mu = 10;
    #
    #gradf0 = [zeros(N,1); ones(N,1)];
    #
    ## starting point --- make sure that it is feasible
    #if (largescale)
    #  if (norm(A(x0)-b)/norm(b) > cgtol)
    #    disp('Starting point infeasible; using x0 = At*inv(AAt)*y.');
    #    AAt = @(z) A(At(z));
    #    [w, cgres, cgiter] = cgsolve(AAt, b, cgtol, cgmaxiter, 0);
    #    if (cgres > 1/2)
    #      disp('A*At is ill-conditioned: cannot find starting point');
    #      xp = x0;
    #      return;
    #    end
    #    x0 = At(w);
    #  end
    #else
    #  if (norm(A*x0-b)/norm(b) > cgtol)
    #    disp('Starting point infeasible; using x0 = At*inv(AAt)*y.');
    #    opts.POSDEF = true; opts.SYM = true;
    #    [w, hcond] = linsolve(A*A', b, opts);
    #    if (hcond < 1e-14)
    #      disp('A*At is ill-conditioned: cannot find starting point');
    #      xp = x0;
    #      return;
    #    end
    #    x0 = A'*w;
    #  end  
    #end
    #x = x0;
    #u = (0.95)*abs(x0) + (0.10)*max(abs(x0));
    #
    ## set up for the first iteration
    #fu1 = x - u;
    #fu2 = -x - u;
    #lamu1 = -1./fu1;
    #lamu2 = -1./fu2;
    #if (largescale)
    #  v = -A(lamu1-lamu2);
    #  Atv = At(v);
    #  rpri = A(x) - b;
    #else
    #  v = -A*(lamu1-lamu2);
    #  Atv = A'*v;
    #  rpri = A*x - b;
    #end
    #
    #sdg = -(fu1'*lamu1 + fu2'*lamu2);
    #tau = mu*2*N/sdg;
    #
    #rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
    #rdual = gradf0 + [lamu1-lamu2; -lamu1-lamu2] + [Atv; zeros(N,1)];
    #resnorm = norm([rdual; rcent; rpri]);
    #
    #pditer = 0;
    #done = (sdg < pdtol) | (pditer >= pdmaxiter);
    #while (~done)
    #  
    #  pditer = pditer + 1;
    #  
    #  w1 = -1/tau*(-1./fu1 + 1./fu2) - Atv;
    #  w2 = -1 - 1/tau*(1./fu1 + 1./fu2);
    #  w3 = -rpri;
    #  
    #  sig1 = -lamu1./fu1 - lamu2./fu2;
    #  sig2 = lamu1./fu1 - lamu2./fu2;
    #  sigx = sig1 - sig2.^2./sig1;
    #  
    #  if (largescale)
    #    w1p = w3 - A(w1./sigx - w2.*sig2./(sigx.*sig1));
    #    h11pfun = @(z) -A(1./sigx.*At(z));
    #    [dv, cgres, cgiter] = cgsolve(h11pfun, w1p, cgtol, cgmaxiter, 0);
    #    if (cgres > 1/2)
    #      disp('Cannot solve system.  Returning previous iterate.  (See Section 4 of notes for more information.)');
    #      xp = x;
    #      return
    #    end
    #    dx = (w1 - w2.*sig2./sig1 - At(dv))./sigx;
    #    Adx = A(dx);
    #    Atdv = At(dv);
    #  else
    #    w1p = -(w3 - A*(w1./sigx - w2.*sig2./(sigx.*sig1)));
    #    H11p = A*(sparse(diag(1./sigx))*A');
    #    opts.POSDEF = true; opts.SYM = true;
    #    [dv,hcond] = linsolve(H11p, w1p, opts);
    #    if (hcond < 1e-14)
    #      disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
    #      xp = x;
    #      return
    #    end
    #    dx = (w1 - w2.*sig2./sig1 - A'*dv)./sigx;
    #    Adx = A*dx;
    #    Atdv = A'*dv;
    #  end
    #  
    #  du = (w2 - sig2.*dx)./sig1;
    #  
    #  dlamu1 = (lamu1./fu1).*(-dx+du) - lamu1 - (1/tau)*1./fu1;
    #  dlamu2 = (lamu2./fu2).*(dx+du) - lamu2 - 1/tau*1./fu2;
    #  
    #  # make sure that the step is feasible: keeps lamu1,lamu2 > 0, fu1,fu2 < 0
    #  indp = find(dlamu1 < 0);  indn = find(dlamu2 < 0);
    #  s = min([1; -lamu1(indp)./dlamu1(indp); -lamu2(indn)./dlamu2(indn)]);
    #  indp = find((dx-du) > 0);  indn = find((-dx-du) > 0);
    #  s = (0.99)*min([s; -fu1(indp)./(dx(indp)-du(indp)); -fu2(indn)./(-dx(indn)-du(indn))]);
    #  
    #  # backtracking line search
    #  suffdec = 0;
    #  backiter = 0;
    #  while (~suffdec)
    #    xp = x + s*dx;  up = u + s*du; 
    #    vp = v + s*dv;  Atvp = Atv + s*Atdv; 
    #    lamu1p = lamu1 + s*dlamu1;  lamu2p = lamu2 + s*dlamu2;
    #    fu1p = xp - up;  fu2p = -xp - up;  
    #    rdp = gradf0 + [lamu1p-lamu2p; -lamu1p-lamu2p] + [Atvp; zeros(N,1)];
    #    rcp = [-lamu1p.*fu1p; -lamu2p.*fu2p] - (1/tau);
    #    rpp = rpri + s*Adx;
    #    suffdec = (norm([rdp; rcp; rpp]) <= (1-alpha*s)*resnorm);
    #    s = beta*s;
    #    backiter = backiter + 1;
    #    if (backiter > 32)
    #      disp('Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)')
    #      xp = x;
    #      return
    #    end
    #  end
    #  
    #  
    #  # next iteration
    #  x = xp;  u = up;
    #  v = vp;  Atv = Atvp; 
    #  lamu1 = lamu1p;  lamu2 = lamu2p;
    #  fu1 = fu1p;  fu2 = fu2p;
    #  
    #  # surrogate duality gap
    #  sdg = -(fu1'*lamu1 + fu2'*lamu2);
    #  tau = mu*2*N/sdg;
    #  rpri = rpp;
    #  rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
    #  rdual = gradf0 + [lamu1-lamu2; -lamu1-lamu2] + [Atv; zeros(N,1)];
    #  resnorm = norm([rdual; rcent; rpri]);
    #  
    #  done = (sdg < pdtol) | (pditer >= pdmaxiter);
    #  
    #  disp(sprintf('Iteration = #d, tau = #8.3e, Primal = #8.3e, PDGap = #8.3e, Dual res = #8.3e, Primal res = #8.3e',...
    #    pditer, tau, sum(u), sdg, norm(rdual), norm(rpri)));
    #  if (largescale)
    #    disp(sprintf('                  CG Res = #8.3e, CG Iter = #d', cgres, cgiter));
    #  else
    #    disp(sprintf('                  H11p condition number = #8.3e', hcond));
    #  end
    #  
    #end
    
    # End of original Matab code
    #----------------------------    
    
    # Nic: check if b is 0; if so, return 0
    #    Otherwise it will break later
    if numpy.linalg.norm(b) < 1e-16:
        return numpy.zeros_like(x0)
    
    #largescale = isa(A,'function_handle');
    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False       
    
    #N = length(x0);
    N = x0.size
    
    alpha = 0.01
    beta = 0.5
    mu = 10
    
    #gradf0 = [zeros(N,1); ones(N,1)];
    gradf0 = numpy.hstack((numpy.zeros(N), numpy.ones(N)))
    
    # starting point --- make sure that it is feasible
    #if (largescale)
    if largescale:
      raise l1eqNotImplementedError('Largescale not implemented yet!')
    else:
      #if (norm(A*x0-b)/norm(b) > cgtol)
      if numpy.linalg.norm(numpy.dot(A,x0)-b) / numpy.linalg.norm(b) > cgtol:
        #disp('Starting point infeasible; using x0 = At*inv(AAt)*y.');
        if verbose:
          print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
        #opts.POSDEF = true; opts.SYM = true;
        #[w, hcond] = linsolve(A*A', b, opts);
        #if (hcond < 1e-14)
        #  disp('A*At is ill-conditioned: cannot find starting point');
        #  xp = x0;
        #  return;
        #end
        #x0 = A'*w;
        try:
            w = scipy.linalg.solve(numpy.dot(A,A.T), b, sym_pos=True)
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
        x0 = numpy.dot(A.T, w)
      #end  
    #end
    x = x0.copy()
    #u = (0.95)*abs(x0) + (0.10)*max(abs(x0));
    u = (0.95)*numpy.abs(x0) + (0.10)*numpy.abs(x0).max()
    
    # set up for the first iteration
    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1/fu1
    lamu2 = -1/fu2
    if (largescale):
      #v = -A(lamu1-lamu2);
      #Atv = At(v);
      #rpri = A(x) - b;
      raise l1eqNotImplementedError('Largescale not implemented yet!')
    else:
      #v = -A*(lamu1-lamu2);
      #Atv = A'*v;
      #rpri = A*x - b;
      v = numpy.dot(-A, lamu1-lamu2)
      Atv = numpy.dot(A.T, v)
      rpri = numpy.dot(A,x) - b
    #end
    
    #sdg = -(fu1'*lamu1 + fu2'*lamu2);
    sdg = -(numpy.dot(fu1,lamu1) + numpy.dot(fu2,lamu2))
    tau = mu*2*N/sdg
    
    #rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
    rcent = numpy.hstack((-numpy.dot(lamu1,fu1),  -numpy.dot(lamu2,fu2))) - (1/tau)
    #rdual = gradf0 + [lamu1-lamu2; -lamu1-lamu2] + [Atv; zeros(N,1)];
    rdual = gradf0 + numpy.hstack((lamu1-lamu2, -lamu1-lamu2)) + numpy.hstack((Atv, numpy.zeros(N)))
    #resnorm = norm([rdual; rcent; rpri]);
    resnorm = numpy.linalg.norm(numpy.hstack((rdual, rcent, rpri)))
    
    pditer = 0
    #done = (sdg < pdtol) | (pditer >= pdmaxiter);
    done = (sdg < pdtol) or (pditer >= pdmaxiter)
    #while (~done)
    while not done:
      
      pditer = pditer + 1
      
      #w1 = -1/tau*(-1./fu1 + 1./fu2) - Atv;
      w1 = -1/tau*(-1/fu1 + 1/fu2) - Atv
      w2 = -1 - 1/tau*(1/fu1 + 1/fu2)
      w3 = -rpri
      
      #sig1 = -lamu1./fu1 - lamu2./fu2;
      sig1 = -lamu1/fu1 - lamu2/fu2
      sig2 = lamu1/fu1 - lamu2/fu2
      #sigx = sig1 - sig2.^2./sig1;
      sigx = sig1 - sig2**2/sig1
      
      if largescale:
        #w1p = w3 - A(w1./sigx - w2.*sig2./(sigx.*sig1));
        #h11pfun = @(z) -A(1./sigx.*At(z));
        #[dv, cgres, cgiter] = cgsolve(h11pfun, w1p, cgtol, cgmaxiter, 0);
        #if (cgres > 1/2)
        #  disp('Cannot solve system.  Returning previous iterate.  (See Section 4 of notes for more information.)');
        #  xp = x;
        #  return
        #end
        #dx = (w1 - w2.*sig2./sig1 - At(dv))./sigx;
        #Adx = A(dx);
        #Atdv = At(dv);
        raise l1eqNotImplementedError('Largescale not implemented yet!')
      else:
        #w1p = -(w3 - A*(w1./sigx - w2.*sig2./(sigx.*sig1)));
        w1p = -(w3 - numpy.dot(A,(w1/sigx - w2*sig2/(sigx*sig1))))
        #H11p = A*(sparse(diag(1./sigx))*A');
        H11p = numpy.dot(A, numpy.dot(numpy.diag(1/sigx),A.T))
        #opts.POSDEF = true; opts.SYM = true;
        #[dv,hcond] = linsolve(H11p, w1p, opts);
        try:
          dv = scipy.linalg.solve(H11p, w1p, sym_pos=True)
          hcond = 1.0/numpy.linalg.cond(H11p)
        except scipy.linalg.LinAlgError:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            return xp
        if hcond < 1e-14:
            if verbose:
              print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
            xp = x.copy()
            return xp            
        #if (hcond < 1e-14)
        #  disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
        #  xp = x;
        #  return
        #end
        
        #dx = (w1 - w2.*sig2./sig1 - A'*dv)./sigx;
        dx = (w1 - w2*sig2/sig1 - numpy.dot(A.T,dv))/sigx
        #Adx = A*dx;
        Adx = numpy.dot(A,dx)
        #Atdv = A'*dv;
        Atdv = numpy.dot(A.T,dv)
      #end
      
      #du = (w2 - sig2.*dx)./sig1;
      du = (w2 - sig2*dx)/sig1
      
      #dlamu1 = (lamu1./fu1).*(-dx+du) - lamu1 - (1/tau)*1./fu1;
      dlamu1 = (lamu1/fu1)*(-dx+du) - lamu1 - (1/tau)*1/fu1
      dlamu2 = (lamu2/fu2)*(dx+du) - lamu2 - 1/tau*1/fu2
      
      # make sure that the step is feasible: keeps lamu1,lamu2 > 0, fu1,fu2 < 0
      #indp = find(dlamu1 < 0);  indn = find(dlamu2 < 0);
      indp = numpy.nonzero(dlamu1 < 0)
      indn = numpy.nonzero(dlamu2 < 0)
      #s = min([1; -lamu1(indp)./dlamu1(indp); -lamu2(indn)./dlamu2(indn)]);
      s = numpy.min(numpy.hstack((numpy.array([1]), -lamu1[indp]/dlamu1[indp], -lamu2[indn]/dlamu2[indn])))
      #indp = find((dx-du) > 0);  indn = find((-dx-du) > 0);
      indp = numpy.nonzero((dx-du) > 0)
      indn = numpy.nonzero((-dx-du) > 0)
      #s = (0.99)*min([s; -fu1(indp)./(dx(indp)-du(indp)); -fu2(indn)./(-dx(indn)-du(indn))]);
      s = (0.99)*numpy.min(numpy.hstack((numpy.array([s]), -fu1[indp]/(dx[indp]-du[indp]), -fu2[indn]/(-dx[indn]-du[indn]))))
      
      # backtracking line search
      suffdec = 0
      backiter = 0
      #while (~suffdec)
      while not suffdec:
        #xp = x + s*dx;  up = u + s*du; 
        xp = x + s*dx
        up = u + s*du 
        #vp = v + s*dv;  Atvp = Atv + s*Atdv; 
        vp = v + s*dv
        Atvp = Atv + s*Atdv
        #lamu1p = lamu1 + s*dlamu1;  lamu2p = lamu2 + s*dlamu2;
        lamu1p = lamu1 + s*dlamu1
        lamu2p = lamu2 + s*dlamu2
        #fu1p = xp - up;  fu2p = -xp - up;  
        fu1p = xp - up
        fu2p = -xp - up
        #rdp = gradf0 + [lamu1p-lamu2p; -lamu1p-lamu2p] + [Atvp; zeros(N,1)];
        rdp = gradf0 + numpy.hstack((lamu1p-lamu2p, -lamu1p-lamu2p)) + numpy.hstack((Atvp, numpy.zeros(N)))
        #rcp = [-lamu1p.*fu1p; -lamu2p.*fu2p] - (1/tau);
        rcp = numpy.hstack((-lamu1p*fu1p, -lamu2p*fu2p)) - (1/tau)
        #rpp = rpri + s*Adx;
        rpp = rpri + s*Adx
        #suffdec = (norm([rdp; rcp; rpp]) <= (1-alpha*s)*resnorm);
        suffdec = (numpy.linalg.norm(numpy.hstack((rdp, rcp, rpp))) <= (1-alpha*s)*resnorm)
        s = beta*s
        backiter = backiter + 1
        if (backiter > 32):
          if verbose:
            print 'Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)'
          xp = x.copy()
          return xp
        #end
      #end
      
      
      # next iteration
      #x = xp;  u = up;
      x = xp.copy()
      u = up.copy()
      #v = vp;  Atv = Atvp; 
      v = vp.copy()
      Atv = Atvp.copy()
      #lamu1 = lamu1p;  lamu2 = lamu2p;
      lamu1 = lamu1p.copy()
      lamu2 = lamu2p.copy()
      #fu1 = fu1p;  fu2 = fu2p;
      fu1 = fu1p.copy()
      fu2 = fu2p.copy()
      
      # surrogate duality gap
      #sdg = -(fu1'*lamu1 + fu2'*lamu2);
      sdg = -(numpy.dot(fu1,lamu1) + numpy.dot(fu2,lamu2))
      tau = mu*2*N/sdg
      rpri = rpp.copy()
      #rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
      rcent = numpy.hstack((-lamu1*fu1, -lamu2*fu2)) - (1/tau)
      #rdual = gradf0 + [lamu1-lamu2; -lamu1-lamu2] + [Atv; zeros(N,1)];
      rdual = gradf0 + numpy.hstack((lamu1-lamu2, -lamu1-lamu2)) + numpy.hstack((Atv, numpy.zeros(N)))
      #resnorm = norm([rdual; rcent; rpri]);
      resnorm = numpy.linalg.norm(numpy.hstack((rdual, rcent, rpri)))
      
      #done = (sdg < pdtol) | (pditer >= pdmaxiter);
      done = (sdg < pdtol) or (pditer >= pdmaxiter)
      
      if verbose:
        print 'Iteration =',pditer,', tau =',tau,', Primal =',numpy.sum(u),', PDGap =',sdg,', Dual res =',numpy.linalg.norm(rdual),', Primal res =',numpy.linalg.norm(rpri)
      if largescale:
        #disp(sprintf('                  CG Res = #8.3e, CG Iter = #d', cgres, cgiter));
        raise l1eqNotImplementedError('Largescale not implemented yet!')
      else:
        #disp(sprintf('                  H11p condition number = #8.3e', hcond));
        if verbose:
          print '                  H11p condition number =',hcond
      #end
      
    #end
    
    return xp