# Performs RHF
import integral_engine as ie
import numpy as np
from scipy.linalg import sqrtm
from Molecule import Molecule

class SCF:
    def __init__(self, geom, basis, charge=0, props=False):
        self.molecule = Molecule(geom,charge)
        self.basis={"name":basis,"functions":ie.initialize(self.molecule,basis)}
        self.ints= self.genInts(properties=props)
        self.E=None
        self.C=None
        self.epsilon=None

    def genInts(self,properties=False):
        """
        Generates the integrals needed for SCF.

        Can optionally compute property integrals (e.g. dipoles)
        as well.
        """
        #Generate list of basis functions
        basis_funcs=self.basis["functions"]

        intDict={}
        #Generate integrals from the basis function list
        intDict['S']=ie.formS(basis_funcs)
        intDict['T']=ie.formT(basis_funcs)
        intDict['V']=ie.formNucAttract(basis_funcs,self.molecule)
        intDict['Pi']=ie.form2e(basis_funcs)

        if properties:
            intDict['mu'] = ie.formMu(basis_funcs)
            intDict['p'] = ie.formP(basis_funcs)
            intDict['L'] = ie.formL(basis_funcs)

        return intDict


    def solve(self,max_iter=100,thresh=1e-8,guess=None):
        """Solves the SCF equations.

        The Fock and density matrices are the total ones, not just the
        alpha spin. This should functionally when passing in a
        guess density.
        """
        T=self.ints['T']
        V=self.ints['V']
        S=self.ints['S']
        Pi=self.ints['Pi']

        # Determine number of occupied orbitals
        Nocc = self.molecule.nelec//2
        #Form core Hamiltonian
        h=T-V

        #Determine orthogonalizing transform
        X=sqrtm(np.linalg.inv(S))

        i=0
        conv=np.inf
        # Form initial density matrix
        if guess is None:
            P_old = np.zeros((len(S),len(S)))
        else:
            P_old=np.array(guess)

        #Calculate nuclear repulsion energy
        E_nuc=self.molecule.getNucRepulsion()

        #SCF loop
        while (i<max_iter and conv>thresh):
            #Form Fock matrix, orthogonalize, diagonalize
            J=np.einsum('ijkl,kl->ij',Pi,P_old)
            K=np.einsum('iljk,kl->ij',Pi,P_old)
            F=(h+J-0.5*K)

            E=np.einsum('ij,ij->',P_old,.5*(F+h))+E_nuc
            F=X.T@F@X
            e,C=np.linalg.eigh(F)
            print(E)

            #Unorthogonalize and form new density matrix
            C=X@C
            P_new=2*np.einsum('pi,qi->pq', C[:,:Nocc], C[:,:Nocc])
            conv=np.sum((P_new-P_old)**2)**.5

            P_old=np.copy(P_new)
            print(f"Current RMSD of Density= {conv}")
            i+=1

        self.E=E
        self.C=C
        self.epsilon=e
        return E,C