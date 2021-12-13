# Performs RHF
import integral_engine as ie
import numpy as np
from scipy.linalg import sqrtm
from Molecule import Molecule

class SCF:
    def __init__(self, geom, basis, charge=0, props=False,solve=True):
        if isinstance(geom, Molecule):
            self._molecule = geom
        else:
            self._molecule = Molecule(geom, charge)
        self._basis={"name":basis, "functions":ie.initialize(self._molecule, basis)}
        self._ints= self.genInts(properties=props)
        self._E=None
        self._C=None
        self._epsilon=None
        if solve:
            self.solve()

    @property
    def molecule(self):
        """Molecule object

        Only readable, better to create a new SCF instance
        if a new molecule is needed.
        """
        return self._molecule

    @property
    def basis(self):
        return self._basis["name"]

    @basis.setter
    def basis(self, basis):
        """Reset basis functions and integrals

        If C is defined, project it onto the new basis.
        """
        old_funcs=self._basis["functions"]
        self._basis={"name":basis, "functions":ie.initialize(self._molecule, basis)}
        self._ints= self.genInts()

        if self._C is not None:
            mixedS=ie.mixedOverlap(self._basis["functions"],old_funcs)
            newS=self._ints["S"]
            newC=np.linalg.inv(newS)@mixedS@self._C
            self._C=newC


    def genInts(self,properties=False):
        """
        Generates the integrals needed for SCF.

        Can optionally compute property integrals (e.g. dipoles)
        as well.
        """
        #Generate list of basis functions
        basis_funcs=self._basis["functions"]

        intDict={}
        #Generate integrals from the basis function list
        intDict['S']=ie.formS(basis_funcs)
        intDict['T']=ie.formT(basis_funcs)
        intDict['V']=ie.formNucAttract(basis_funcs, self._molecule)
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
        T=self._ints['T']
        V=self._ints['V']
        S=self._ints['S']
        Pi=self._ints['Pi']

        # Determine number of occupied orbitals
        Nocc = self._molecule.nelec // 2
        #Form core Hamiltonian
        h=T-V

        #Determine orthogonalizing transform
        X=sqrtm(np.linalg.inv(S))

        i=0
        conv=np.inf
        # Form initial density matrix
        if guess is not None:
            P_old = np.array(guess)
        elif self._C is not None:
            P_old = 2*np.einsum('pi,qi->pq', self._C[:,:Nocc], self._C[:,:Nocc])
        else:
            P_old = np.zeros((len(S), len(S)))

        #Calculate nuclear repulsion energy
        E_nuc=self._molecule.getNucRepulsion()

        #SCF loop
        while (i<max_iter and conv>thresh):
            #Form Fock matrix, orthogonalize, diagonalize
            J=np.einsum('ijkl,kl->ij',Pi,P_old)
            K=np.einsum('iljk,kl->ij',Pi,P_old)
            F=(h+J-0.5*K)

            E=np.einsum('ij,ij->',P_old,.5*(F+h))+E_nuc
            F=X.T@F@X
            e,C=np.linalg.eigh(F)
            print(f"Current energy:{E}")

            #Unorthogonalize and form new density matrix
            C=X@C
            P_new=2*np.einsum('pi,qi->pq', C[:,:Nocc], C[:,:Nocc])
            conv=np.sum((P_new-P_old)**2)**.5

            P_old=np.copy(P_new)
            print(f"Current RMSD of Density= {conv}")
            i+=1

        print("----------------------------------------------------")
        print(f"SCF converged in {i} iterations")
        print(f"SCF energy= {E}")
        print(f"SCF RMSD of Density= {conv}")
        self._E=E
        self._C=C
        self._epsilon=e
        return E,C