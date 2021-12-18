import pytest
import ez_elec
import numpy as np

class TestMolecule():
    @pytest.fixture()
    def molList(self):
        geom = [['H', 0.0, 0.0, -0.375],
                ['H', 0.0, 0.0, 0.375],
                ['H', 0.0, -0.375, 0.0],
                ['H', 0.0, 0.375, 0.0]]
        charge=1
        units="ang"
        return ez_elec.Molecule(geom,charge=charge,units=units)

    @pytest.fixture()
    def molString(self):
        geom = """
               O 0.0 0.0 0.0
               H 0.0 1.0 -1.0
               H 0.0 -1.0 -1.0
               """
        return ez_elec.Molecule(geom)

    def test_atomList(self,molList):
        assert molList.atoms==["H","H","H","H"]

    def test_coordsList(self,molList):
        coord=ez_elec.TO_BOHR["ang"]*np.array([[0.0, 0.0, -0.375],[0.0, 0.0, 0.375],
                                            [0.0, -0.375, 0.0],[0.0, 0.375, 0.0]])
        np.testing.assert_allclose(molList.coords,coord)

    def test_charge(self,molList):
        assert molList.charge==1

    def test_elecCount(self,molList):
        assert molList.nelec==3
        # Make total number of electrons negative
        with pytest.raises(ValueError):
            molList.charge=5

    def test_atomsList(self,molString):
        assert molString.atoms==["O","H","H"]

    def test_coordsString(self,molString):
        coord=(ez_elec.TO_BOHR["ang"]
               *np.array([[0.0, 0.0, 0.0],[0.0, 1.0, -1.0],[0.0, -1.0, -1.0]]))
        np.testing.assert_allclose(molString.coords,coord)
