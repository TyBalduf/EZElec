import numpy as np
from periodictable import elements

#Constants
TO_BOHR={'bohr':1,'nm':1.8897261245e1,'pm':1.8897261245e-2,'ang':1.8897261245}
##Dictionary of element atomic numbers
ATOMIC_CHARGES={elem.symbol:elem.number for elem in elements}
##Dictionary of element atomic masses
ATOMIC_MASSES={elem.symbol:elem.mass for elem in elements}

class Molecule:
    def __init__(self, geom, charge=0, units="ang"):
        atoms,coords=self._convertTo(geom,units=units)
        self._atoms=atoms
        self._coords=coords
        self._charge=charge
        self._nelec=self.elecCount()
        self._units=units

    @property
    def atoms(self):
        return self._atoms

    @property
    def coords(self):
        return self._coords

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self,value):
        self._charge=value
        self._nelec=self.elecCount()

    @property
    def nelec(self):
        return self._nelec

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self,value):
        self._units=value

    @staticmethod
    def _convertTo(geom,units="ang"):
        """Converts the geometry from a string or nested list to a tuple of atoms and coordinates"""
        atoms = []
        coords = []

        #Handle input string, each line is a separate atom
        if isinstance(geom,str):
            geom=geom.strip("\n")
            geom = [g.split() for g in geom.split('\n')]

        for atom in geom:
            elem, *coord = atom
            atoms.append(elem)
            coords.append(coord)
        coords=np.array(coords,dtype=float)*TO_BOHR[units]
        return atoms, coords

    def _convertFrom(self):
        """Converts the molecule from a tuple of atoms and coordinates to a nested list"""
        geom = []
        for atom,coord in zip(self.atoms,self.coords):
            temp=[atom]
            temp.extend(coord/TO_BOHR[self.units])
            geom.append(temp)
        return geom

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, item):
        return self.atoms[item], self.coords[item]

    def __str__(self):
        return str([(a,coord/TO_BOHR[self.units]) for a,coord in self])

    def __copy__(self):
        geom=self._convertFrom()
        return Molecule(geom, self.charge, units=self.units)

    def getCharges(self, chargeValues = ATOMIC_CHARGES):
        """Returns a list of the charges of the atoms in the molecule"""
        charges = [chargeValues[a] for a in self.atoms]
        return charges

    def elecCount(self):
        """Returns the number of electrons in a molecule"""
        count = sum(self.getCharges())-self.charge
        if count < 0:
            raise ValueError(f"A charge of {self.charge} leads to a negative"
                             " number of electrons")
        return count

    def getNucRepulsion(self):
        """Returns the nuclear repulsion energy of the molecule"""
        E_nuc = 0
        charges = self.getCharges()
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                E_nuc += charges[i]*charges[j]/np.linalg.norm(self.coords[i]-self.coords[j])
        return E_nuc

    def replaceAtom(self,atom,index):
        """Replaces an atom in the molecule"""
        self._atoms[index]=atom

    def translate(self,vector,indices=None):
        """Translates the molecule (or individual atoms) by a vector"""
        if indices is None:
            self._coords+=vector
        elif isinstance(indices,int):
            self._coords[indices]+=vector
        else:
            for i in indices:
                self._coords[i]+=vector

    def rotate(self,axis,angle):
        """Rotates the molecule around an axis by an angle"""
        self.coords = self.coords@makeU(axis,angle).T
        return self

def makeU(axis,angle):
    """Make a 3x3 rotation matrix around a particular axis

    parameters:
    axis: the axis to rotate around (array-like of len 3)
    angle: the angle to rotate by (in degrees)
    """
    ##normalize
    axis=np.array(axis,dtype=float)
    norm=np.dot(axis,axis)**.5
    axis/=norm

    ##form U
    angle=np.radians(angle)
    cross=np.array([[0,-axis[2],axis[1]],
                    [axis[2],0,-axis[0]],
                    [-axis[1],axis[0],0]])
    s=np.sin(angle)
    c=np.cos(angle)
    U=(c*np.eye(3)+s*cross+(1-c)*np.outer(axis,axis))
    ##remove small values
    U[abs(U)<1e-10]=0
    return U
