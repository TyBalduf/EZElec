#Class to create basis function objects
import numpy as np
from functools import lru_cache
from scipy.special import factorial2 as fac2


#Constants
ANG_TO_BOHR = 1.8897261245

class Basis_Function:
    def __init__(self,center,coeffs,exponents,shell):
        self.center=np.array(center)*ANG_TO_BOHR
        self.coeffs=np.array(coeffs)
        self.exps=np.array(exponents)
        self.shell=shell
        self.norms=None
        self.N=None
        self.normalize()

    def normalize(self):
        '''Return norms of the primitives and N that
        will normalize the contracted Gaussian'''
        order=np.sum(self.shell)
        facts=np.prod(fac2([2*s-1 for s in self.shell]))
        self.norms=(((2/np.pi)**(3/4))*(2**order)*(facts**(-1/2))*
                     self.exps**((2*order+3)/4))

        pre=(np.pi**1.5)*facts*(2.0**-order)
        divisor=np.add.outer(self.exps,self.exps)**-(order+1.5)
        normalized=self.norms*self.coeffs
        summand=(pre*np.einsum('i,j,ij->',normalized,normalized,divisor))**-.5
        self.N=summand

    def overlap(self,other):
        '''Determine overlap between contracted Gaussians via
        Obara-Saika recurrence'''
        distance=self.center-other.center

        value=0
        #Loop over each contraction
        for na,ca,ea in zip(self.norms,self.coeffs,self.exps):
            for nb,cb,eb in zip(other.norms,other.coeffs,other.exps):
                temp=1
                #Determine x/y/z overlaps based on the shell
                for x,i,j in zip(distance,self.shell,other.shell):
                    temp*=oneElec_ObSa(i,j,x,ea,eb)
                value+=na*ca*nb*cb*temp
        return self.N*other.N*value


#Utility functions
@lru_cache(maxsize=64)
def oneElec_ObSa(i,j,x,alpha,beta):
    '''Obara Saika recurrence for one electron overlap.
    Overlap integrals between Gaussians with different angular
    momentum are related. Can generate higher angular momentum
    from the base case of s-type overlap.'''

    p=alpha+beta
    mu=(alpha*beta)/p

    #Base cases
    if i==0 and j==0:
        return np.sqrt(np.pi/p)*np.exp(-mu*x**2)
    elif i<0 or j<0:
        return 0.0
    elif i>0:
        upper=(-beta/p)*x*oneElec_ObSa(i-1,j,x,alpha,beta)
        lower=((i-1)*oneElec_ObSa(i-2,j,x,alpha,beta)+
               j*oneElec_ObSa(i-1,j-1,x,alpha,beta))/(2*p)
        return upper+lower
    elif j>0:
        upper=(alpha/p)*x*oneElec_ObSa(i,j-1,x,alpha,beta)
        lower=(i*oneElec_ObSa(i-1,j-1,x,alpha,beta)+
               (j-1)*oneElec_ObSa(i,j-2,x,alpha,beta))/(2*p)
        return upper+lower
