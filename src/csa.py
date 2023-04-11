from scipy.linalg import orth
def topk(mat,k):
# select top k eigenvectors(normalized) and eigenvalues
    e_vals,e_vecs = np.linalg.eig(mat)
    sorted_indices = np.argsort(e_vals)
    return e_vals[sorted_indices[:-k-1:-1]],e_vecs[:,sorted_indices[:-k-1:-1]]

def CSA(X,m1,m2,T_max=4,eps=0.001):
# X: N*w*h
# m1,m2: dimention after reductoin m1*m2
# T_max: max iterations
# eps: break condition
# output: U1,U2,status(1 for convergence and 0 for else)
    t=1
    N = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    S1=[h,w]
    S2=[m1,m2]
    while 1: # initialize projection matrix
        U1=np.random.randn(h,m1);
        U1=orth(U1)
        if U1.shape[1]==m1:
            break
    while 1: # initialize projection matrix
        U2=np.random.randn(w,m2);
        U2=orth(U2)
        if U2.shape[1]==m2:
            break
    U_new_0=U_old_0=U1
    U_new_1=U_old_1=U2
    while t<T_max:
        C1=np.zeros((S1[0],S1[0]))
        C2=np.zeros((S1[1],S1[1]))
        for i in range(N):
            Xi=X[i]
            X1_X1=np.dot(Xi,U_new_1)
            C1=C1+np.dot(X1_X1,X1_X1.T)
        ev,U_new_0=topk(C1,S2[0]) # update U0
        for i in range(N):
            Xi=X[i]
            X1_X1=np.dot(U_new_0.T,Xi)
            C2=C2+np.dot(X1_X1.T,X1_X1)
        ev,U_new_1=topk(C2,S2[1]) # update U1
        # break condition
        if t>=2 and np.trace(np.abs(np.dot(U_new_0.T,U_old_0)))/S2[0]>1-eps:
            if np.trace(np.abs(np.dot(U_new_1.T,U_old_1)))/S2[1]>1-eps:
                return U_new_0,U_new_1,1
        U_old_0=U_new_0
        U_old_1=U_new_1
        t=t+1
    return U_new_0,U_new_1,0