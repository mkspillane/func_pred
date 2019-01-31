from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt



# Inputs: Time series (x_i,y_i)
# Want to predict z = y_{i+1}
# Model structure: Use x to predict the linear coefficients for y which are then used to predict using a linear model. 
# This should work well for functions which are solutions to linear ODE's where the coefficients are constant or linear in x.  
# Examples include sin(x), exp(x), exp(-x^2), Airy functions.  
# We could include polynomial coeffients by changing the X matrix.  However, this is rather ad hoc and hopefully a NN can
# be used to allow for non polynomial functions of x.  




def forward_X(X,A,B):
    # X is shape (m,n) where m is the number of examples and n is the number of input points
    # A is shape (n+1,n)
    # B is shape (n+1,)
    
    m=X.shape[0]
    n=X.shape[1]
    
    
    temp = np.dot(X,A.transpose())+B.transpose()
    a = temp[:,:-1]
    b= temp[:,-1]
    return a,b
    
def forward_Y(Y,a,b):
    # Y is shape (m,n) where m is the number of examples and n is the number of input points
    # a is the first output of forward_X
    # b is the second output of forward_X
    m=Y.shape[0]
    z = np.sum(a*Y,axis=1)+b
    z=z.reshape(m,1)
    return z
    
def grad(Z,X,Y,A,B):
    # this calculates the cost and gradients of the cost wrt coeffiencts in A and B
    # gradients are outputs them as dA and dB with the same dimensions as well as cost 
    
    
    m=X.shape[0]
    n=X.shape[1]
    


    temp_a,temp_b = forward_X(X,A,B)
    z_hat = forward_Y(Y,temp_a,temp_b)
    cost = 1.*np.sum((z_hat-Z)**2)/m #the cost function is same as used for linear regression. Change if needed for your particular problem


    da = -2./m*(Y*(Z-z_hat)).reshape((m,n))
    db = -2./m*(Z-z_hat).reshape((m,1))
    dalpha = np.concatenate((da,db),axis=1)

    dB = np.sum(dalpha,axis=0).reshape((n+1,1))
    dA = np.einsum('ij,ik->kj',X,dalpha)
    
    return cost, dA,dB
    
def grad_desc(Z,X,Y,A_s,B_s,n_epoch,epoch_per,learn_rate):
    # standard gradient descent
    A_t=A_s
    B_t=B_s
    J_hist = np.zeros(n_epoch)
    m=X.shape[0]
    epoch_size = np.floor(m*epoch_per).astype(int)
    k = m/epoch_size
    for i in range(n_epoch):
        cost, dA,dB =grad(Z,X,Y,A_t,B_t)
        J_hist[i] = cost
        for j in range(k):

            cost, dA,dB =grad(Z[epoch_size*j:epoch_size*(j+1),:],X[epoch_size*j:epoch_size*(j+1),:],Y[epoch_size*j:epoch_size*(j+1),:],A_t,B_t)
            A_t -= learn_rate*dA
            B_t -= learn_rate*dB
    
    return J_hist,A_t,B_t

    
def adam_grad(Z,X,Y,A_s,B_s,n_epoch,epoch_per,learning_rate,beta1=.9,beta2=.999,epsilon=10.**-8):
    #adam gradient decent
    A_t=A_s
    B_t=B_s
    J_hist = np.zeros(n_epoch)
    m=X.shape[0]
    epoch_size = np.floor(m*epoch_per).astype(int)
    k = m/epoch_size
    mw_A = 0*A
    mw_B = 0*B
    vw_A=0*A
    vw_B=0*B
    for i in range(n_epoch):
        cost, dA,dB =grad(Z,X,Y,A_t,B_t)
        J_hist[i] = cost
        for j in range(k):

            cost, dA,dB =grad(Z[epoch_size*j:epoch_size*(j+1),:],X[epoch_size*j:epoch_size*(j+1),:],Y[epoch_size*j:epoch_size*(j+1),:],A_t,B_t)
            mw_A = beta1*mw_A+(1-beta1)*dA
            mw_B = beta1*mw_B+(1-beta1)*dB
            vw_A = beta2*vw_A+(1-beta2)*(dA**2)
            vw_B = beta2*vw_B+(1-beta2)*(dB**2)
            
            mA = mw_A/(1-beta1**(i+1))
            vA = vw_A/(1-beta2**(i+1))
            
            mB = mw_B/(1-beta1**(i+1))
            vB = vw_B/(1-beta2**(i+1))
            
            A_t -= learning_rate*mA/(np.sqrt(vA)+epsilon)
            B_t -= learning_rate*mB/(np.sqrt(vB)+epsilon)
    
    return J_hist,A_t,B_t   
    
    
# Below is a test case.  exp(-x^2) is a solution to the differential equation y'+2xy=0.  The sample is drawn only from x<0 
# but the prediction at the end is for x>0.  The model does a good job predicting future values of y as can be seen in the plot.
    
examples = 50000
points_in = 2
divisor = 10.


np.random.seed(1)
X1 = np.zeros((examples,points_in))
Y1 = np.zeros((examples,points_in))
Z1= np.zeros((examples,1))

def f(x,sigma):
    # The function to be learned.
    
    return np.sin(-x/2)

s=4.

for i in range(examples):
    rand = 20*(np.random.rand()-1)
    X1_temp = 1.*np.arange(points_in).reshape(-1, 1)/divisor+rand
    X1_temp = X1_temp.reshape((points_in,))
    X2_temp = f(X1_temp,s).ravel()
    X2_temp = X2_temp.reshape((points_in,))
    Y1[i,:] = X2_temp
    X1[i,:] = X1_temp
    
    Y1_temp = 1.*(np.arange(1).reshape(-1, 1)+points_in)/divisor+rand
    Y2_temp = (f(Y1_temp,s)-f(Y1_temp-1/divisor,s)).ravel()
    Y2_temp = Y2_temp.reshape((1,))
    Z1[i,:] = Y2_temp
  
np.random.seed(2)
A=np.random.randn(points_in+1,points_in)*0.1
B=2*np.random.rand(points_in+1,1)-1


J,Af,Bf = adam_grad(Z1,X1,Y1,A,B,600,.1,.002)




pred_x = np.arange(100).reshape(-1, 1)/divisor
act_f = f(pred_x,s)
pred_f = 0*np.sin(2 * np.pi * pred_x)


for i in range(points_in):
    pred_f[i]=act_f[i]


Xp = np.concatenate((pred_x,act_f,pred_f),axis=1)

for i in range(Xp.shape[0]-points_in):
    a_temp,b_temp = forward_X(Xp[i:i+points_in,0].reshape((1,points_in)),Af,Bf)
    Xp[i+points_in,2]=forward_Y(Xp[i:i+points_in,2].reshape((1,points_in)),a_temp,b_temp)+Xp[i+points_in-1,2]
    
plt.plot(Xp[:,0],Xp[:,1],'r',Xp[:,0],Xp[:,2],'b')
plt.show()


    