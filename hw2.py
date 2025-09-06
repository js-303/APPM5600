import numpy as np
import scipy.linalg
from scipy.linalg import hilbert
from numpy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt


H_matrices = {}
I_matrices = {}
RHS_matrices = {}
GE_x = {}
GE_errnorm = {}
GE_resnorm = {}
GS_x = {}
GD_x = {}
GS_errnorm = {}
GS_resnorm = {}
E_matrices = {}
F_matrices = {}
D_matrices = {}
x_0s = {}
p = {}

def GS_2(D,E,F,b,x_0,k):
    x_k = x_0
    for l in range(k):
        x_k1 = inv(D-E)@F@x_k+inv(D-E)@b
        x = x_k1
    return x

def G_S(A, b, x_0, k):
    """
    Run k steps of Gauss_Seidel method
    A: the matrix
    b: the right-hand-side
    x_0: initial guess x0
    k: the number of steps
    """ 
    # Get the size of the system
    n = A.shape[0]
    # Initialize the solution vector
    x = x_0.copy()
    
    for l in range(k):
        for i in range(n):
            sum = 0.
            for j in range(n):            
                if i != j:
                    sum += A[i,j]*x[j]
            x[i] = (b[i]-sum)/A[i,i]   
    return x



for i in range(2,14):
    H_matrices[i] = np.array(hilbert(i))
    I_matrices[i] = np.ones(i)
    RHS_matrices[i] = np.array(hilbert(i))@np.ones(i)
    E_matrices[i], D_matrices[i], p[i] = scipy.linalg.ldl(H_matrices[i], lower=True)
    F_matrices[i] = E_matrices[i].T
    print(norm((E_matrices[i]@D_matrices[i]@F_matrices[i])-H_matrices[i]))
    GE_x[i] = np.linalg.solve(H_matrices[i], RHS_matrices[i])
    GS_x[i] = G_S(H_matrices[i], RHS_matrices[i], x_0=np.zeros(i), k=1000)
    #GS_x[i] = GS_2(D_matrices[i],E_matrices[i],F_matrices[i],RHS_matrices[i],x_0=np.zeros(i),k=1400)
    GE_errnorm[i] = norm(np.ones(i)-GE_x[i], ord=np.inf)
    GS_errnorm[i] = norm(np.ones(i)-GS_x[i], ord=np.inf)
    GE_resnorm[i] = norm(RHS_matrices[i]-H_matrices[i]@GE_x[i], ord=np.inf)
    GS_resnorm[i] = norm(RHS_matrices[i]-H_matrices[i]@GS_x[i], ord=np.inf)

n = []
for key in GS_errnorm:
    n = list(GS_errnorm.keys())
print(n)
for value in GS_errnorm.values():
    print(value)

fig1, (ax1,ax2) = plt.subplots(2)
ax1.plot([key for key in GE_errnorm.keys()], [value for value in GE_errnorm.values()])
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_title('GE error norm')
ax2.plot([key for key in GS_errnorm.keys()], [value for value in GS_errnorm.values()])
ax2.set_yscale("log")
ax2.set_xscale("linear")
ax2.set_title('GS error norm')

fig2, (ax1,ax2) = plt.subplots(2)
ax1.plot([key for key in GE_resnorm.keys()], [value for value in GE_resnorm.values()])
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_title('GE residual norm')
ax2.plot([key for key in GS_resnorm.keys()], [value for value in GS_resnorm.values()])
ax2.set_yscale("log")
ax2.set_xscale("linear")
ax2.set_title('GS residual norm')

plt.tight_layout()
plt.show()


    






#print(H_matrices,"\n")
#print(I_matrices)
#print(RHS_matrices,"\n")
#print(GE_x)
#print(GS_x)
#print(GE_errnorm)
#print(GS_errnorm)
#print(GE_resnorm)
#print(GS_resnorm)
#print(E_matrices[5])
#print(F_matrices[5])
#print(D_matrices[5])


