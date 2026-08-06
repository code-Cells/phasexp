from phasexp.ff import ForceField

def test_from_itp():
    top = ForceField.from_itp("/usr/local/gromacs/share/gromacs/top/charmm36-jul2022.ff/ffbonded.itp")
