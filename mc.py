from .molecule import Molecule
import numpy as np

def mcmc(
        conformation: Molecule, 
        # new_conformation: Molecule=None, 
        temp: float=298.15, 
        n_steps: int=100):
    kbt = kb * temp
    for _ in range(n_steps):
        proposal = conformation.transform(x)
        delta_e = proposal.energy() - conformation.energy()
        if metropolis(delta_e, kbt):
            conformation = proposal

def metropolis(delta_e: float, kbt: float) -> bool:
    if delta_e <= 0:
        return True
    p = np.exp(-delta_e / kbt)
    return np.random.rand() < p
