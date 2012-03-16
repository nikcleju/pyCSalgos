
import numpy
import cvxopt
import cvxopt.solvers
import cvxopt.msk
import mosek

def cvxopt_lp(y, A):

    N = A.shape[1]
    AA = numpy.hstack((A, -A))
    c = numpy.ones((2*N, 1))
    
    G = numpy.vstack((numpy.zeros((2*N,2*N)), -numpy.eye(2*N)))
    h = numpy.zeros((4*N,1))

    # Convert numpy arrays to cvxopt matrices
    cvx_c = cvxopt.matrix(c)
    cvx_G = cvxopt.matrix(G)
    cvx_h = cvxopt.matrix(h)
    cvx_AA = cvxopt.matrix(AA)
    cvx_y = cvxopt.matrix(y.reshape(y.size,1))
    
    # Options    
    cvxopt.solvers.options['show_progress'] = False
    #cvxopt.solvers.options['MOSEK'] = {mosek.iparam.log: 0}
    
    # Solve
    #res = cvxopt.solvers.lp(cvx_c, cvx_G, cvx_h, A=cvx_AA, b=cvx_y, solver='mosek')
    res = cvxopt.solvers.lp(cvx_c, cvx_G, cvx_h, A=cvx_AA, b=cvx_y)
      
    primal = numpy.squeeze(numpy.array(res['x']))
    gamma = primal[:N] - primal[N:]
    return gamma
    
    #lb = zeros(2*N,1);
    #ub = Inf*ones(2*N,1);
    ##[primal,obj,exitflag,output2,dual] = linprog(c,[],[],A,y,lb,ub,[],opt);
    ##[primal,~,~,~,~] = linprog(c,[],[],A,aggy,lb,ub);
    #[primal,obj,exitflag,output2,dual] = linprog(c,[],[],A,aggy,lb,ub);
    #gamma = primal(1:N) - primal((N+1):(2*N));