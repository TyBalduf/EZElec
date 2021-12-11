import numpy as np

#Constants
TO_BOHR={'bohr':1,'nm':1.8897261245e1,'pm':1.8897261245e-2,'ang':1.8897261245}
##Dictionary of element atomic numbers
_ATOMIC_CHARGES={"H":1,"He":2,
                  "Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10}

class Molecule:
    def __init__(self, geom, charge,units="ang"):
        atoms,coords=self._convert(geom,units=units)
        self.atoms=atoms
        self.coords=coords
        self.charge=charge
        self.nelec=self.elecCount()

    @staticmethod
    def _convert(geom,units="ang"):
        """Converts the geometry to a list of tuples"""
        atoms = []
        coords = []
        for atom in geom:
            elem, *coord = atom
            atoms.append(elem)
            coords.append(np.array(coord)*TO_BOHR[units])
        return atoms, coords

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, item):
        return self.atoms[item], self.coords[item]

    def getCharges(self, chargeValues = _ATOMIC_CHARGES):
        """Returns a list of the charges of the atoms in the molecule"""
        charges = [chargeValues[a] for a in self.atoms]
        return charges

    def elecCount(self):
        """Returns the number of electrons in a molecule"""
        count = sum(self.getCharges())-self.charge
        return count

    def getNucRepulsion(self):
        """Returns the nuclear repulsion energy of the molecule"""
        E_nuc = 0
        charges = self.getCharges()
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                E_nuc += charges[i]*charges[j]/np.linalg.norm(self.coords[i]-self.coords[j])
        return E_nuc