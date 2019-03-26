
import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy



def initialize_parameters_deep_tl(layer_dims,rint=3):

    np.random.seed(rint)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        
        parameters['Wt' + str(l)] = .01*np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['bt' + str(l)] = np.zeros((layer_dims[l], 1))
        parameters['Wl' + str(l)] = .01*np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['bl' + str(l)] = np.zeros((layer_dims[l], 1))
        

        
    return parameters
    
def forward_layer_tl(A_prev,Wt,Wl,bt,bl):
    Q = np.dot(Wt,A_prev)+bt
    A = np.tanh(Q)+np.dot(Wl,A_prev)+bl
    
    cache = (Q,A,np.tanh(Q))
    
    return A, cache
    
    
def NN_forward(X,Y,parameters):
    
    m = Y.shape[1]
    L = len(parameters)//4
    
    cache = {'A0': X}
    
    A = X
    
    for l in range(1,L+1):
        A_prev = A
        A,temp_cache =forward_layer_tl(A_prev, parameters['Wt' + str(l)],parameters['Wl' + str(l)],parameters['bt' + str(l)],parameters['bl' + str(l)])
        
        cache['Q' + str(l)] = temp_cache[0]
        cache['A' + str(l)] = temp_cache[1]
        cache['tQ' + str(l)] = temp_cache[2]
    
    Z = np.sum(A*Y,axis=0).reshape((1,m))
    
    return Z,cache
    
    

    
    

def backward_layer_tl(Wt,Wl,bt,bl,Q,A,tQ,dA_next):
    
    cache={}
        
    dbl = np.sum(dA_next,axis=1)
    dWl = np.dot(dA_next,A.T)
    dWt = np.dot(((1-tQ**2)*dA_next),A.T)
    dbt = np.sum((1-tQ**2)*dA_next,axis=1)
    
    
    dA = np.dot(Wl.T,dA_next)+np.dot(Wt.T,dA_next*(1-tQ**2))
    
    cache['dbl'] = dbl
    cache['dWl'] = dWl
    cache['dbt'] = dbt
    cache['dWt'] = dWt
    
    return dA, cache
    

    
def NN_back_tl(X,Y,Z,parameters):
    
    L = len(parameters)//4
    
    grad = {}
    
    m= Z.shape[1]
    
    Z_hat,cache = NN_forward(X,Y,parameters)
    
    #cost = np.sum((Z-Z_hat)**2,axis=1)/m
    #dcost = 2*Y*(Z_hat-Z)/m
    delta = 10**-4
    cost = np.sum(np.log(delta+(Z_hat-Z)**2)-np.log(delta),axis = 1)/m
    dcost = ((2*(Z_hat-Z))/((Z_hat-Z)**2+delta))*Y/m
    
    dA = dcost
    
    for l in range(L,0,-1):
        
        dA,cache_t = backward_layer_tl(parameters['Wt' + str(l)],parameters['Wl' + str(l)],parameters['bt' + str(l)],parameters['bl' + str(l)],cache['Q' + str(l)],cache['A' + str(l-1)],cache['tQ' + str(l)],dA)
        
        grad['dbl'+str(l)] = cache_t['dbl'].reshape(parameters['bl' + str(l)].shape)
        grad['dWl'+str(l)] = cache_t['dWl']
        grad['dbt'+str(l)] = cache_t['dbt'].reshape(parameters['bt' + str(l)].shape)
        grad['dWt'+str(l)] = cache_t['dWt']
        
        
    return cost, grad
    
    
def mom_grad_desc(Z,X,Y,parameters,n_epoch,epoch_per,learn_rate,beta1 = .9):
    # standard gradient descent
    

    
    L = len(parameters)//4
    

    
    J_hist = np.zeros(n_epoch)
    para_temp = copy.deepcopy(parameters)
    m=X.shape[1]
    epoch_size = np.floor(m*epoch_per).astype(int)

    k = np.floor(m/epoch_size)
    k= k.astype(int)

    para_g={}
    for l in range(1, L+1):

        para_g['vWt' + str(l)]=0*parameters['Wt' + str(l)]
        para_g['vbt' + str(l)]=0*parameters['bt' + str(l)]
        para_g['vWl' + str(l)]=0*parameters['Wl' + str(l)]
        para_g['vbl' + str(l)]=0*parameters['bl' + str(l)]
        
        
    for i in range(n_epoch):
        cost,_ = NN_back_tl(X,Y,Z,para_temp)
        J_hist[i] = cost
        for j in range(k):
            cost, grad_t =NN_back_tl(X[:,epoch_size*j:epoch_size*(j+1)],Y[:,epoch_size*j:epoch_size*(j+1)],Z[:,epoch_size*j:epoch_size*(j+1)],para_temp)

            for l in range(1, L+1):

                para_g['vWt' + str(l)]=beta1*para_g['vWt' + str(l)]+(1-beta1)*grad_t['dWt' + str(l)]
                para_g['vbt' + str(l)]=beta1*para_g['vbt' + str(l)]+(1-beta1)*grad_t['dbt' + str(l)]
                para_g['vWl' + str(l)]=beta1*para_g['vWl' + str(l)]+(1-beta1)*grad_t['dWl' + str(l)]
                para_g['vbl' + str(l)]=beta1*para_g['vbl' + str(l)]+(1-beta1)*grad_t['dbl' + str(l)]
                
                para_temp['Wl'+str(l)] -= learn_rate*para_g['vWl' + str(l)]
                para_temp['bl'+str(l)] -= learn_rate*para_g['vbl' + str(l)]
                para_temp['Wt'+str(l)] -= learn_rate*para_g['vWt' + str(l)]
                para_temp['bt'+str(l)] -= learn_rate*para_g['vbt' + str(l)]
            
            
    return J_hist, para_temp
    
    
def adam_grad(Z,X,Y,parameters,n_epoch,epoch_per,learn_rate,beta1=.9,beta2=.999,epsilon=10.**-8):
    #adam gradient decent
    
    L = len(parameters)//4
    
    J_hist = np.ones(n_epoch)
    para_temp = copy.deepcopy(parameters)
    para_best = copy.deepcopy(parameters)
    m=X.shape[1]
    epoch_size = np.floor(m*epoch_per).astype(int)

    k = np.floor(m/epoch_size)
    k= k.astype(int)

    para_g={}
    for l in range(1, L+1):

        para_g['vWt' + str(l)]=0*parameters['Wt' + str(l)]
        para_g['vbt' + str(l)]=0*parameters['bt' + str(l)]
        para_g['vWl' + str(l)]=0*parameters['Wl' + str(l)]
        para_g['vbl' + str(l)]=0*parameters['bl' + str(l)]
        
        para_g['mWt' + str(l)]=0*parameters['Wt' + str(l)]
        para_g['mbt' + str(l)]=0*parameters['bt' + str(l)]
        para_g['mWl' + str(l)]=0*parameters['Wl' + str(l)]
        para_g['mbl' + str(l)]=0*parameters['bl' + str(l)]

    for i in range(n_epoch):
        cost,_ = cost,_ = NN_back_tl(X,Y,Z,para_temp)
        if cost<J_hist.min():
            para_best = copy.deepcopy(para_temp)
        else:
            pass
        J_hist[i] = cost
        for j in range(k):

            cost, grad_t = NN_back_tl(X[:,epoch_size*j:epoch_size*(j+1)],Y[:,epoch_size*j:epoch_size*(j+1)],Z[:,epoch_size*j:epoch_size*(j+1)],para_temp)
            for l in range(1, 4):
                para_g['mWt' + str(l)]=beta1*para_g['mWt' + str(l)]+(1-beta1)*grad_t['dWt' + str(l)]
                para_g['mbt' + str(l)]=beta1*para_g['mbt' + str(l)]+(1-beta1)*grad_t['dbt' + str(l)]
                para_g['mWl' + str(l)]=beta1*para_g['mWl' + str(l)]+(1-beta1)*grad_t['dWl' + str(l)]
                para_g['mbl' + str(l)]=beta1*para_g['mbl' + str(l)]+(1-beta1)*grad_t['dbl' + str(l)]
                
                
                para_g['vWt' + str(l)]=beta2*para_g['vWt' + str(l)]+(1-beta2)*(grad_t['dWt' + str(l)]**2)
                para_g['vbt' + str(l)]=beta2*para_g['vbt' + str(l)]+(1-beta2)*(grad_t['dbt' + str(l)]**2)
                para_g['vWl' + str(l)]=beta2*para_g['vWl' + str(l)]+(1-beta2)*(grad_t['dWl' + str(l)]**2)
                para_g['vbl' + str(l)]=beta2*para_g['vbl' + str(l)]+(1-beta2)*(grad_t['dbl' + str(l)]**2)
                
                
                para_temp['Wt'+str(l)] -= learn_rate*np.sqrt(1-beta2**(i+1))/(1-beta1**(i+1))*para_g['mWt' + str(l)]/(np.sqrt(para_g['vWt' + str(l)])+epsilon)
                para_temp['bt'+str(l)] -= learn_rate*np.sqrt(1-beta2**(i+1))/(1-beta1**(i+1))*para_g['mbt' + str(l)]/(np.sqrt(para_g['vbt' + str(l)])+epsilon)
                para_temp['Wl'+str(l)] -= learn_rate*np.sqrt(1-beta2**(i+1))/(1-beta1**(i+1))*para_g['mWl' + str(l)]/(np.sqrt(para_g['vWl' + str(l)])+epsilon)
                para_temp['bl'+str(l)] -= learn_rate*np.sqrt(1-beta2**(i+1))/(1-beta1**(i+1))*para_g['mbl' + str(l)]/(np.sqrt(para_g['vbl' + str(l)])+epsilon)
            
    
    return J_hist,para_temp,para_best

    
def f(x,sigma):
    # The function to be learned.
    
    return np.exp(-x**2/4)*x

    
examples = 80000
points_in = 2
divisor = 10.
s=4

np.random.seed(1)
X1 = np.zeros((examples,points_in))
Y1 = np.zeros((examples,points_in))
Z1= np.zeros((examples,1))
    
    
for i in range(examples):
    rand = 20*(np.random.rand()-1)
    X1_temp = 1.*np.arange(points_in).reshape(-1, 1)/divisor+rand
    X1_temp = X1_temp.reshape((points_in,))
    X2_temp = f(X1_temp,s).ravel()+np.random.randn()*0.0
    X2_temp = X2_temp.reshape((points_in,))
    Y1[i,:] = X2_temp
    X1[i,:] = X1_temp
    
    Y1_temp = 1.*(np.arange(1).reshape(-1, 1)+points_in)/divisor+rand
    Y2_temp = (f(Y1_temp,s)-f(Y1_temp-1/divisor,s)).ravel()
    Y2_temp = Y2_temp.reshape((1,))
    Z1[i,:] = Y2_temp
    
X1= X1.T
Y1=Y1.T
Z1=Z1.T
    
para = initialize_parameters_deep_tl((points_in,3,3,points_in),rint=3)   
Para = initialize_parameters_deep_tl((points_in,3,3,points_in),rint=4)   



j,_,para_f=adam_grad(Z1,X1,Y1,para,500,.1,.001)
J,_,Para_f=adam_grad(Z1,X1,Y1,Para,500,.1,.001)

j2,para_f2=mom_grad_desc(Z1,X1,Y1,para_f,1500,.1,.001)
J2,Para_f2=mom_grad_desc(Z1,X1,Y1,Para_f,1500,.1,.001)


pred_x = np.arange(100).reshape(-1, 1)/divisor
act_f = f(pred_x,s)
pred_f = 0*np.sin(2 * np.pi * pred_x)
pred_f2 = 0*np.sin(2 * np.pi * pred_x)


for i in range(points_in):
    pred_f[i]=act_f[i]
    pred_f2[i]=act_f[i]

Xp = np.concatenate((pred_x,act_f,pred_f,pred_f2),axis=1)

for i in range(Xp.shape[0]-points_in):
    Xp[i+points_in,2]=NN_forward(Xp[i:i+points_in,0].reshape((points_in,1)),Xp[i:i+points_in,2].reshape((points_in,1)),para_f2)[0]+Xp[i+points_in-1,2]
    Xp[i+points_in,3]=NN_forward(Xp[i:i+points_in,0].reshape((points_in,1)),Xp[i:i+points_in,3].reshape((points_in,1)),Para_f2)[0]+Xp[i+points_in-1,3]
   
    
    
plt.plot(Xp[:,0],Xp[:,1],'r',Xp[:,0],Xp[:,2],'b',Xp[:,0],Xp[:,2],'g')
plt.show()
