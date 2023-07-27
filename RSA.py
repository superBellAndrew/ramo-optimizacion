import numpy as np
from mealpy.optimizer import Optimizer

class RSA(Optimizer):
    def __init__(self,epoch=1000, pop_size=100,alpha=0.1, beta=0.1, **kwargs):

        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.eps = 2e-14
        self.set_parameters(["epoch", "pop_size", "alpha", "beta"])
    
    def bounded_position(self, position=None, lb=None, ub=None):
        for i in range(position.shape[0]):
            for j in range(position.shape[1]):
                if position[i,j] <lb[j] or position[i,j]>ub[j]:
                    position[i,j] = np.random.random()*(ub[j]-lb[j]) + lb[j]
        return position

    
    def M(self,x):
        return (np.sum(x,axis=1)/x.shape[1]).reshape(-1,1)

    def P(self,x,best,LB,UB,alpha,eps):
        return alpha*( x-self.M(x) ) / ( best*(UB-LB)+eps )

    def Eta(self,best,p):
        return best*p

    def R(self,best,x,eps):
        N = x.shape[0]
        index = np.random.randint(0,N,size=(N,))
        x_r2 = x[index]
        return (best-x_r2)/(best+eps)

    def ES(self,t,T):
        return 2 * np.random.normal() * (1-t/T)

    def evolve(self, epoch):

        best_x = self.g_best[0]    
        x = np.array( [ pop_i[0] for pop_i in self.pop ] )


        p = self.P(x,best_x,self.problem.lb,self.problem.ub,self.alpha,self.eps)
        eta = self.Eta(best_x,p)                                                             # EQ4
        r = self.R(best_x,x,self.eps)                                                        # EQ5
        es = self.ES(epoch,self.epoch)                                                                         # EQ6
        rand = np.random.random(size=(self.pop_size,self.problem.n_dims))

        if epoch <= self.epoch/4:                        # HIGH WALKING
            x = best_x - eta*self.beta - r*rand
        
        elif epoch<=(2*self.epoch/4) and epoch>(self.epoch/4):          # BELLY WALKING
            index = np.random.randint(0,self.pop_size,size=(self.pop_size,))
            x_r1 = x[index]
            x = best_x*x_r1*es*rand
        
        elif epoch<=(3*self.epoch/4) and epoch>(2*self.epoch/4):      # HUNTING CORDINATION
            x = best_x*p*rand
        
        else:                               # HUNTING COOPERATION
            x = best_x-eta*self.eps-r*rand
        
        x = self.bounded_position(x,self.problem.lb, self.problem.ub)
        x = [ [xi,[self.problem.fit_func(xi),[self.problem.fit_func(xi),]]] for xi in x ]
        self.pop = x


