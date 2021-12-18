#Class to create basis function objects
import numpy as np
from functools import lru_cache
from scipy.special import factorial2 as fac2
from scipy.special import gamma,gammainc
import typing

#Static typing variables
Ang_mom=typing.Tuple[int,int,int]
Dist=typing.Tuple[float,float,float]
Exp_2e=typing.Tuple[float,float,float,float]
Dist_2e=typing.Tuple[Dist,Dist,Dist]
Ang_2e=typing.Tuple[Ang_mom,Ang_mom,Ang_mom,Ang_mom]

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
        """Set norms of the primitives and N of the contracted Gaussian"""

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
        """Determine overlap between contracted Gaussians via Obara-Saika recurrence.

         Can do standard overlap, as well as derivatives (specify deriv)
         and multipoles (specify deriv and set multi True for the desired components)
        """

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
        """Evaluate electron-nuclear attraction primitive integrals

        Uses the Coulomb variant of the Obara-Saika recurrence.
        """
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

#Integral Recursions
@lru_cache()
def ObSa_1e(alpha:float, beta:float, x:float, i=0, j=0, t=0, x_c=None)->float:
    """Obara Saika recurrence for one electron overlap.

    Can generate higher angular momentum from the base
    case of s-type overlap.

    :param alpha: bra exponent
    :param beta: ket exponent
    :param x: Distance between bra and ket
    :param i: Angular momentum of bra
    :param j: Angular momentum of ket
    :param t: Derivative/multipole order
    :param x_c: Center of charge (for multipole)
    :return: value of overlap integral
    """

    p=alpha+beta
    mu=(alpha*beta)/p

    #Base cases
    if i==0 and j==0 and t==0:
        return np.sqrt(np.pi/p)*np.exp(-mu*x**2)
    elif i<0 or j<0 or t<0:
        return 0.0
    #Reduce order of derivative/multipole, creates
    #terms with increased and decreased bra momentum
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
    #Decrease bra momentum
    elif i>0:
        upper= (-beta/p) * x * ObSa_1e(alpha, beta, x, i - 1, j, t, x_c)
        lower= ((i-1) * ObSa_1e(alpha, beta, x, i - 2, j, t, x_c) +
                j * ObSa_1e(alpha, beta, x, i - 1, j - 1, t, x_c))/(2*p)
        if x_c is None:
            lower+= -2*beta*t*ObSa_1e(alpha, beta, x, i-1, j, t-1)/(2*p)
        else:
            lower+= t*ObSa_1e(alpha, beta, x, i-1, j, t-1, x_c)/(2*p)
        return upper+lower
    #Decrease ket momentum
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
def ObSa_Coulomb(p: int, mu: int, bra_dist: Dist, ket_dist: Dist, nuc_dist: Dist,
                 bra_ang: Ang_mom = (0,0,0), ket_ang: Ang_mom = (0,0,0), N: int = 0)->float:
    """Evaluate one-electron Coulomb attraction

    Uses the Obara-Saika recursion relations to convert angular momentum
    to higher orders of the Boys function.

    :param p: Sum of bra-ket exponents
    :param mu: Product of bra-ket exponents, divided by p
    :param bra_dist: Distance from bra to combined Gaussian center
    :param bra_ang: Angular momentum of bra
    :param ket_dist: Distance from ket to combined Gaussian center
    :param ket_ang: Angular momentum of ket
    :param nuc_dist: Distance between nuclei associated with bra and ket
    :param N: Boys' function order
    """

    ##Accumulate as angular momentum is converted to Boy's order N
    value=0

    if bra_ang==ket_ang==(0,0,0):
        pre=(2*np.pi/p)
        K=np.prod(np.exp(-mu*(np.array(bra_dist)-np.array(ket_dist))**2))
        distance=np.einsum('i,i->',nuc_dist,nuc_dist)
        boy=Boys(N,p*distance)
        value+= pre*K*boy
    elif any(b<0 for b in bra_ang) or any(k<0 for k in ket_ang):
        value+=0
    else:
        ##Loop over x/y/z, reducing the angular momentum
        for x in range(3):
            ##Reduce bra momentum first, then ket
            args = [p, mu, bra_dist, ket_dist, nuc_dist]
            if bra_ang[x]>0:
                bra_temp=tuple(b-1 if x==t else b for t,b in enumerate(bra_ang))
                value+=(bra_dist[x]*ObSa_Coulomb(*args,bra_temp,ket_ang,N)
                       -nuc_dist[x]*ObSa_Coulomb(*args,bra_temp,ket_ang,N+1))

                ket_temp=tuple(b-1 if x==t else b for t,b in enumerate(ket_ang))
                value+=(ket_ang[x]) * (ObSa_Coulomb(*args,bra_temp,ket_temp,N)
                         -ObSa_Coulomb(*args,bra_temp,ket_temp,N+1))/(2*p)

                bra_temp = tuple(b - 1 if x == t else b for t, b in enumerate(bra_temp))
                value+=(bra_ang[x]-1)*(ObSa_Coulomb(*args,bra_temp,ket_ang,N)
                       -ObSa_Coulomb(*args,bra_temp,ket_ang,N+1))/(2*p)
                break

            elif ket_ang[x]>0:
                ket_temp = tuple(b - 1 if x == t else b for t, b in enumerate(ket_ang))
                value+=(ket_dist[x]*ObSa_Coulomb(*args,bra_ang,ket_temp,N)
                       -nuc_dist[x]*ObSa_Coulomb(*args,bra_ang,ket_temp,N+1))

                bra_temp = tuple(b - 1 if x == t else b for t, b in enumerate(bra_ang))
                value+=(bra_ang[x]) * (ObSa_Coulomb(*args,bra_temp,ket_temp,N)
                         -ObSa_Coulomb(*args,bra_temp,ket_temp,N+1))/(2*p)

                ket_temp = tuple(b - 1 if x == t else b for t, b in enumerate(ket_temp))
                value+=(ket_ang[x]-1)*(ObSa_Coulomb(*args,bra_ang,ket_temp,N)
                       -ObSa_Coulomb(*args,bra_ang,ket_temp,N+1))/(2*p)
                break
    return value

@lru_cache(maxsize=4096)
def ObSa_2e(expon: Exp_2e,dists: Dist_2e,momentums: Ang_2e,N=0)->float:
    """Evaluates two electron integrals using simplified Obara-Saika

    Recursion relations adapted from Chapter 12, Eqs. 238-241 of Modern
    Electronic Structure Theory.

    :param expon: Exponent of each orbital, left to right, chemist notation
    :param dists: Should contain: distance from bra center to 1st orbital [0],
                  distance from  ket center to 3rd orbital [1], and distance between
                  the bra and ket centers [2].
    :param momentums: Momentum of each orbital, left to right, chemist notation
    :param N: Boys' function order
    :return: Value of two electron integral
    """

    value=0
    ##Unpack exponents
    a, b, c, d = expon
    p = a + b
    q = c + d

    #Base case
    if all(ang==(0,0,0) for ang in momentums):
        ##Form necessary exponent factors
        mu=a*b/p
        nu=c*d/q
        omega=p*q/(p+q)

        ##Construct base case
        pre=(2*np.pi**2.5)/(p*q*(p+q)**.5)
        K_ab=np.prod(np.exp(-mu*((p/b)*np.array(dists[0]))**2))
        K_cd=np.prod(np.exp(-nu*((q/d)*np.array(dists[1]))**2))
        R_sq=np.einsum('i,i->',dists[2],dists[2])
        boy=Boys(N,omega*R_sq)
        value+= pre*K_ab*K_cd*boy

    elif any(a<0 for ang in momentums for a in ang):
        value+=0

    else:
        #Loop over x/y/z, reducing the angular momentum
        orb_i, orb_j, orb_k, orb_l = momentums
        for x in range(3):
            #Transfer momentum from orbital j/l to orbital i/k
            if orb_j[x]>0:
                minus_j=tuple(b-1 if x == t else b for t, b in enumerate(orb_j))
                plus_i=tuple(b+1 if x == t else b for t, b in enumerate(orb_i))

                transfer=(plus_i,minus_j,orb_k,orb_l)
                reduce=(orb_i,minus_j,orb_k,orb_l)
                value+=ObSa_2e(expon,dists,transfer,N=N)
                value+=-(p/b)*dists[0][x]*ObSa_2e(expon,dists,reduce,N=N)
                break

            elif orb_l[x]>0:
                minus_l=tuple(b-1 if x == t else b for t, b in enumerate(orb_l))
                plus_k=tuple(b+1 if x == t else b for t, b in enumerate(orb_k))

                transfer=(orb_i,orb_j,plus_k,minus_l)
                reduce=(orb_i,orb_j,orb_k,minus_l)
                value+=ObSa_2e(expon, dists, transfer,N=N)
                value+=-(q/d)*dists[1][x]*ObSa_2e(expon, dists, reduce,N=N)
                break

            #Tranfer momentum from elec 2 to elec 1 (i.e. from orb_k to orb_i)
            elif orb_k[x]>0:
                minus_k = tuple(b-1 if x == t else b for t, b in enumerate(orb_k))
                min2_k = tuple(b-2 if x == t else b for t, b in enumerate(orb_k))
                plus_i = tuple(b+1 if x == t else b for t, b in enumerate(orb_i))
                minus_i = tuple(b-1 if x == t else b for t, b in enumerate(orb_i))

                transfer=(plus_i,orb_j,minus_k,orb_l)
                reduce=(orb_i,orb_j,minus_k,orb_l)
                double_reduce=(orb_i,orb_j,min2_k,orb_l)
                both_reduce=(minus_i,orb_j,minus_k,orb_l)
                value+=(p*dists[0][x]+q*dists[1][x])*ObSa_2e(expon,dists,reduce,N=N)
                value+=(orb_i[x]/2)*ObSa_2e(expon,dists,both_reduce,N=N)
                value+=(minus_k[x]/2)*ObSa_2e(expon,dists,double_reduce,N=N)
                value+=-p*ObSa_2e(expon,dists,transfer,N=N)
                value/=q
                break

            #Convert remaining angular momentum into higher order Boys functions
            elif orb_i[x]>0:
                omega=p*q/(p+q)
                minus_i = tuple(b - 1 if x == t else b for t, b in enumerate(orb_i))
                min2_i = tuple(b - 2 if x == t else b for t, b in enumerate(orb_i))

                reduce=(minus_i,orb_j,orb_l,orb_k)
                double_reduce=(min2_i,orb_j,orb_l,orb_k)
                value+=dists[0][x]*ObSa_2e(expon,dists,reduce,N=N)
                value+=-(omega*dists[2][x]/p)*ObSa_2e(expon,dists,reduce,N=N+1)
                value+=(minus_i[x]/(2*p))*ObSa_2e(expon,dists,double_reduce,N=N)
                value+=-(minus_i[x]*omega/(2*p**2))*ObSa_2e(expon,dists,double_reduce,N=N+1)
                break

    return value


def Boys(n:int,x:float)->float:
    """Evaluates the Boys' functions

    Expressed in terms of the incomplete Gamma function.

    :param n: order
    :param x: position
    """
    if abs(x)<1e-15:
        Fn=1/(2*n+1)
    else:
        Fn=.5*gamma(n+.5)*gammainc(n+.5,x)*(x**-(n+.5))
    return Fn


class ChargeDist:
    """Class for storing charge distributions

    Convenient for handling two electron integrals in chemist notation.
    """

    def __init__(self,first: Basis_Function,second: Basis_Function,indices: typing.Tuple[int,int]):
        self.first=first
        self.second=second
        self.indices=indices

    def interact(self,other:"ChargeDist"):
        value = 0
        # Loop over each contraction
        for na, ca, ea in zip(self.first.norms, self.first.coeffs, self.first.exps):
            for nb, cb, eb in zip(self.second.norms, self.second.coeffs, self.second.exps):
                P = (ea * self.first.center + eb * self.second.center)/(ea + eb)
                PA= tuple(P-self.first.center)
                for nc, cc, ec in zip(other.first.norms, other.first.coeffs, other.first.exps):
                    for nd, cd, ed in zip(other.second.norms, other.second.coeffs, other.second.exps):
                        Q = (ec*other.first.center+ed*other.second.center)/(ec + ed)
                        QC= tuple(Q-other.first.center)
                        PQ= tuple(P-Q)

                        exponents=(ea,eb,ec,ed)
                        distances=(PA,QC,PQ)
                        momentum=(self.first.shell,self.second.shell,
                                  other.first.shell,other.second.shell)

                        value+=(na*nb*nc*nd)*(ca*cb*cc*cd)*ObSa_2e(exponents,distances,momentum)
                        #print(ObSa_2e.cache_info())

        return (self.first.N*self.second.N)*(other.first.N*other.second.N)*value

