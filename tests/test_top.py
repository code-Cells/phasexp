from .testtools import clock
from phasexp.ff import ForceField

@clock
def test_from_itp():
    top = ForceField.from_itp("/home/eg/charmm36-jul2022.ff/ffbonded.itp")
