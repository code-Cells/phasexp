from .molecule import Molecule
from .constants import kb
from numpy.typing import NDArray
import numpy as np

def mcmc(
        conformation: Molecule,
        vec: NDArray,
        cutoff: 5, 
        temp: float=298.15, 
        n_steps: int=100):
    kbt = kb * temp
    conf_e = conformation.energy(cutoff)
    for _ in range(n_steps):
        proposal = conformation.transform(vec)
        delta_e = proposal.energy(cutoff) - conf_e
        if metropolis(delta_e, kbt):
            conformation = proposal
    return conformation, conf_e

def metropolis(delta_e: float, kbt: float) -> bool:
    if delta_e <= 0:
        return True
    p = np.exp(-delta_e / kbt)
    return np.random.rand() < p
