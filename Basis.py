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

    def overlap(self,other,deriv=(0,0,0),multi=(False,False,False)):
        '''Determine overlap between contracted Gaussians via
        Obara-Saika recurrence. Can do standard overlap, as well as
        derivatives (specify deriv) and multipoles (specify deriv and
        charge center in multi'''
        distance=self.center-other.center

        value=0
        #Loop over each contraction
        for na,ca,ea in zip(self.norms,self.coeffs,self.exps):
            for nb,cb,eb in zip(other.norms,other.coeffs,other.exps):
                temp=1
                P=(ea*self.center+eb*other.center)/(ea+eb)
                x_c=[P[i] if multi[i] else None for i in range(3)]

                #Determine x/y/z overlaps based on the shell
                for vals in zip(distance,self.shell,other.shell,deriv,x_c):
                    temp*=ObSa_1e(ea, eb, *vals)
                value+=na*ca*nb*cb*temp
        return self.N*other.N*value



#Utility functions
@lru_cache()
def ObSa_1e(alpha, beta, x, i=0, j=0, t=0, x_c=None):
    '''Obara Saika recurrence for one electron overlap.
    Can generate higher angular momentum from the base
    case of s-type overlap.'''

    p=alpha+beta
    mu=(alpha*beta)/p

    #Base cases
    if i==0 and j==0 and t==0:
        return np.sqrt(np.pi/p)*np.exp(-mu*x**2)
    elif i<0 or j<0 or t<0:
        return 0.0
    elif t>0:
        if x_c is None:
            upper= 2 * alpha * ObSa_1e(alpha, beta, x, i + 1, j, t-1)
            lower= -i * ObSa_1e(alpha, beta, x, i-1, j, t-1)
        else:
            upper= x_c*ObSa_1e(alpha, beta, x, i, j, t-1, x_c)
            lower=(i * ObSa_1e(alpha, beta, x, i-1, j, t-1, x_c) +
                   j * ObSa_1e(alpha, beta, x, i, j-1, t-1, x_c) +
                   (t-1) * ObSa_1e(alpha, beta, x, i, j, t-2, x_c))/(2*p)
        return upper+lower
    elif i>0:
        upper= (-beta/p) * x * ObSa_1e(alpha, beta, x, i - 1, j, t, x_c)
        lower= ((i-1) * ObSa_1e(alpha, beta, x, i - 2, j, t, x_c) +
                j * ObSa_1e(alpha, beta, x, i - 1, j - 1, t, x_c))/(2*p)
        if x_c is None:
            lower+= -2*beta*t*ObSa_1e(alpha, beta, x, i-1, j, t-1)/(2*p)
        else:
            lower+= t*ObSa_1e(alpha, beta, x, i-1, j, t-1, x_c)/(2*p)
        return upper+lower
    elif j>0:
        upper= (alpha/p) * x * ObSa_1e(alpha, beta, x, i, j-1, t, x_c)
        lower= (i * ObSa_1e(alpha, beta, x, i - 1, j - 1, t, x_c) +
               (j-1) * ObSa_1e(alpha, beta, x, i, j - 2, t, x_c))/(2*p)
        if x_c is None:
            lower+= 2*alpha*t*ObSa_1e(alpha, beta, x, i, j-1, t-1)/(2*p)
        else:
            lower+= t*ObSa_1e(alpha, beta, x, i, j-1, t-1, x_c)/(2*p)
        return upper+lower

