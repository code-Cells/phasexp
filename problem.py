from __future__ import annotations
from .mc import mcmc
from .molecule import Molecule
from pymoo.core.problem import Problem
import numpy as np
import periodictable as pt


class PhaseExplorarionProblem(Problem):
        def __init__(self, molecule: Molecule, n_obj):
            self.molecule = molecule
            super().__init__(
                n_var=10,
                n_obj=n_obj,
                xl=0,
                xu=1,
                vtype=int)

        def _evaluate(self, x, out, *args, **kwargs):
            conformation = Molecule.transform(x)
            energy = mcmc(conformation)
            out["F"] = energy


def run(mol: Molecule):
    masses = [getattr(pt, e).mass for e in mol.elements]
    return masses
