import pytest
import ez_elec
import numpy as np

class TestMolecule_List():
    @pytest.fixture()
    def mol(self):
        geom = [['H', 0.0, 0.0, -0.375],
                ['H', 0.0, 0.0, 0.375],
                ['H', 0.0, -0.375, 0.0],
                ['H', 0.0, 0.375, 0.0]]
        charge=1
        units="ang"
        return ez_elec.Molecule(geom,charge=charge,units=units)

    def test_coords(self,mol):
        coord=ez_elec.TO_BOHR["ang"]*np.array([[0.0, 0.0, -0.375],[0.0, 0.0, 0.375],
                                            [0.0, -0.375, 0.0],[0.0, 0.375, 0.0]])
        np.testing.assert_allclose(mol.coords,coord)

    def test_charge(self,mol):
        assert mol.charge==1