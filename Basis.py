#Class to create basis function objects
import numpy as np
from functools import lru_cache
from scipy.special import factorial2 as fac2
from scipy.special import gamma,gammainc


class Basis_Function:
    def __init__(self,center,coeffs,exponents,shell):
        self.center=np.array(center)
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
        return

    def overlap(self,other,deriv=(0,0,0),multi=(False,False,False)):
        '''Determine overlap between contracted Gaussians via
        Obara-Saika recurrence. Can do standard overlap, as well as
        derivatives (specify deriv) and multipoles (specify deriv and set
        multi True for the desired components)'''
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

    def Coulomb_1e(self,other,nuc_center):
        value = 0
        nuc_center=np.array(nuc_center)
        # Loop over each contraction
        for na, ca, ea in zip(self.norms, self.coeffs, self.exps):
            for nb, cb, eb in zip(other.norms, other.coeffs, other.exps):
                P = (ea*self.center+eb*other.center)/(ea + eb)

                ##Distances
                PA=tuple(P-self.center)
                PB=tuple(P-other.center)
                PC=tuple(P-nuc_center)

                ##Exponent factors
                p=ea+eb
                mu=ea*eb/p


                temp=ObSa_Coulomb(p,mu,PA,PB,PC,self.shell,other.shell)
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

@lru_cache()
def ObSa_Coulomb(p,mu,bra_dist,ket_dist,nuc_dist,
                 bra_ang=(0,0,0),ket_ang=(0,0,0),N=0):

    ##Accumulate as angular momentum is converted to Boy's order N
    value=0

    if bra_ang==ket_ang==(0,0,0):
        pre=(2*np.pi/p)
        K=np.prod(np.exp(-mu*(np.array(bra_dist)-np.array(ket_dist))**2))
        distance=np.einsum('i,i->',nuc_dist,nuc_dist)
        boy=Boys(N,p*distance)
        value+= pre*K*boy
    elif any(b<0 for b in bra_ang):# or any(k<0 for k in ket_ang):
        value+=0
    else:
        ##Loop over x/y/z, reducing the angular momentum
        for i in range(3):
            ##Reduce bra momentum first, then ket
            args = [p, mu, bra_dist, ket_dist, nuc_dist]
            if bra_ang[i]>0:
                bra_temp=list(bra_ang).copy()
                bra_temp[i]-=1
                value+=(bra_dist[i]*ObSa_Coulomb(*args,tuple(bra_temp),ket_ang,N)
                       -nuc_dist[i]*ObSa_Coulomb(*args,tuple(bra_temp),ket_ang,N+1))

                ket_temp=list(ket_ang).copy()
                ket_temp[i]-=1
                ket_temp=tuple(ket_temp)
                value+=(ket_ang[i]) * (ObSa_Coulomb(*args,tuple(bra_temp),ket_temp,N)
                         -ObSa_Coulomb(*args,tuple(bra_temp),ket_temp,N+1))/(2*p)

                bra_temp[i]-=1
                bra_temp=tuple(bra_temp)
                value+=(bra_ang[i]-1)*(ObSa_Coulomb(*args,bra_temp,ket_ang,N)
                       -ObSa_Coulomb(*args,bra_temp,ket_ang,N+1))/(2*p)

            elif ket_ang[i]>0:
                ket_temp=list(ket_ang).copy()
                ket_temp[i]-=1
                value+=(ket_dist[i]*ObSa_Coulomb(*args,bra_ang,tuple(ket_temp),N)
                       -nuc_dist[i]*ObSa_Coulomb(*args,bra_ang,tuple(ket_temp),N+1))

                bra_temp=list(bra_ang).copy()
                bra_temp[i]-=1
                bra_temp=tuple(bra_temp)
                value+=(bra_ang[i]) * (ObSa_Coulomb(*args,bra_temp,tuple(ket_temp),N)
                         -ObSa_Coulomb(*args,bra_temp,tuple(ket_temp),N+1))/(2*p)

                ket_temp[i]-=1
                ket_temp=tuple(ket_temp)
                value+=(ket_ang[i]-1)*(ObSa_Coulomb(*args,bra_ang,ket_temp,N)
                       -ObSa_Coulomb(*args,bra_ang,ket_temp,N+1))/(2*p)

    return value


def Boys(n,x):
    if (abs(x)<1e-15):
        Fn=1/(2*n+1)
    else:
        Fn=.5*gamma(n+.5)*gammainc(n+.5,x)*(x**-(n+.5))
    return Fn