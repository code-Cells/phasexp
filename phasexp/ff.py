from __future__ import annotations
from numbers import Number
from pathlib import Path
from typing import Any, Tuple
import networkx as nx
import numpy as np
import traceback


class FileFormatError(Exception): pass
class InternalCoordinateError(Exception): pass


class ForceField():
    def __init__(self):
        self.atom_types = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.graph = nx.Graph()

    @staticmethod
    def _invalid_line(line: str) -> bool:
        if line.startswith(';') or line.startswith('[') or line.startswith('#') or len(line.strip().replace("\n", '')) == 0:
            return True
        return False

    @staticmethod
    def from_itp(filepath: str, max_depth: int=0, depth: int=0) -> ForceField:
        top = ForceField()
        if depth > max_depth+1:
            return top
        with open(filepath) as f:
            raw = f.readlines()
        for line in raw:
            if line.startswith("#include") and line.endswith(".itp\"\n"):
                child_path = Path(filepath).parent/line.split('"')[1]
                child_top = ForceField.from_itp(child_path, max_depth=max_depth, depth=depth+1)
                top += child_top
        # if len(includes) > 0:
            # for include in includes:
            #     top += include
            # return top 
        sections = []
        for i, line in enumerate(raw):
            if line.startswith("[ "):
                if line[2:].startswith('atomtypes'):
                    sections.append(("At", i))
                if line[2:].startswith('bondtypes'):
                    sections.append(("B", i))
                elif line[2:].startswith('angletypes'):
                    sections.append(("A", i))
                elif line[2:].startswith('dihedraltypes'):
                    sections.append(("D", i))
                elif line[2:].startswith('pairtypes'):
                    sections.append(("P", i))
        sections.append(("", i))
        try:
            for i, (s, j) in enumerate(sections[:-1]):
                k = sections[i+1][1]
                for line in raw[j+1:k-1]:
                    if ForceField._invalid_line(line):
                        continue
                    if s == "At":
                        ForceField._parse_atom_line(top, line)
                    elif s == "B":
                        ForceField._parse_bond_line(top, line)
                    elif s == "A":
                        ForceField._parse_angle_line(top, line)
                    elif s == "D":
                        ForceField._parse_dihedral_line(top, line)
        except:
            traceback.print_exc()
            raise ValueError(f"In file {filepath}, line {line}")
        blacklist = set()
        for bond_par in top.bonds:
            index = bond_par[0], bond_par[1]
            for i in index:
                if i not in blacklist:
                    top.graph.add_node(i)
                    blacklist.add(i)
        return top

    @staticmethod
    def _parse_atom_line(top: ForceField, line: str):
        u = line[:7].strip()
        i = int(line[7:13].strip())
        m = float(line[13:26].strip())
        c = float(line[26:35].strip())
        sigma = float(line[42:57].strip())
        epsilon = float(line[57:68].strip())
        atom_par = AtomParemeter(u, i, m, c, sigma, epsilon)
        top.atom_types[u] = atom_par

    @staticmethod
    def _parse_bond_line(top: ForceField, line: str):
        u = line[:9].strip()
        v = line[9:18].strip()
        func = int(line[18:24])
        b0 = float(line[24:37])
        kb = float(line[37:50])
        bond_par = BondParemeter(u, v, func, b0, kb)
        top.bonds[_sort_tuple(u, v)] = bond_par

    @staticmethod
    def _parse_angle_line(top: ForceField, line: str):
        u = line[:9].strip()
        v = line[9:18].strip()
        w = line[18:27].strip()
        func = int(line[27:33])
        theta = float(line[33:46])
        ktheta = float(line[46:59])
        ub = float(line[59:72])
        kub = float(line[72:85])
        angle_par = AngleParemeter(u, v, w, func, theta, ktheta, ub, kub)
        top.angles[_sort_tuple(u, v, w)] = angle_par

    @staticmethod
    def _parse_dihedral_line(top: ForceField, line: str):
        u = line[:9].strip()
        v = line[9:18].strip()
        w = line[18:27].strip()
        x = line[27:36].strip()
        func = int(line[36:42])
        phi = float(line[42:55])
        kphi = float(line[55:68])
        try:
            mult = int(line[68:74])
        except Exception:
            # traceback.print_exc()
            mult = None
        dihedral_par = DihedralParemeter(u, v, w, x, func, phi, kphi, mult)
        top.dihedrals[_sort_tuple(u, v, w, x)] = dihedral_par

    def __str__(self) -> str:
        out = f"{[i for i in self.graph.nodes()]}\n"
        out += f"{[str(i) for i in self.bonds]}\n"
        out += f"{[str(i) for i in self.angles]}\n"
        out += f"{[str(i) for i in self.dihedrals]}"
        return out

    def __add__(self, other: ForceField) -> ForceField:
        ff = ForceField()
        ff.atom_types = self.atom_types | other.atom_types
        ff.bonds = self.bonds | other.bonds
        ff.angles = self.angles | other.angles
        ff.dihedrals = self.dihedrals | other.dihedrals
        ff.graph = nx.compose(self.graph, other.graph)
        return ff

    def __iadd__(self, other: ForceField):
        self.atom_types |= other.atom_types
        self.bonds |= other.bonds
        self.angles |= other.angles
        self.dihedrals |= other.dihedrals
        self.graph = nx.compose(self.graph, other.graph)
        return self

    def atom(self, a: str):
        return self.atom_types[a]

    def bond(self, a: str, b: str):
            return self.atom_types[(a, b)]

    def angle(self, a: str, b: str, c: str):
            return self.atom_types[(a, b, c)]

    def dihedral(self, a: str, b: str, c: str, d: str):
            return self.atom_types[(a, b, c, d)]
    

class AtomParemeter():
    def __init__(self, 
            u: str, 
            i: int, 
            m: float, 
            c: float, 
            sigma: float, 
            epsilon: float):
        self.u = u
        self.i = i
        self.m = m
        self.c = c
        self.sigma = sigma
        self.epsilon = epsilon

    def __str__(self) -> str:
        return f"{self.u}, {self.i}, {self.m}, {self.c}, {self.sigma}, {self.epsilon}"


class BondParemeter():
    def __init__(self, 
            u: str, 
            v: str, 
            func: int, 
            b: float, 
            kb: float):
        self.u = u
        self.v = v
        self.func = func
        self.b = b
        self.kb = kb

    def __str__(self) -> str:
        return f"{self.u}, {self.v}, {self.func}, {self.b}, {self.kb}"


class AngleParemeter():
    def __init__(self, 
            u: str, 
            v: str, 
            w: str,
            func: int, 
            theta: float, 
            ktheta: float,
            ub: float,
            kub: float):
        self.u = u
        self.v = v
        self.w = w
        self.func = func
        self.theta = theta
        self.ktheta = ktheta
        self.ub = ub
        self.kub = kub

    def __str__(self) -> str:
        return f"{self.u}, {self.v}, {self.w}, {self.func}, {self.theta}, {self.ktheta}, {self.ub}"


class DihedralParemeter():
    def __init__(self, 
            u: str, 
            v: str, 
            w: str,
            x: str,
            func: int, 
            phi: float, 
            kphi: float,
            mult: float):
        self.u = u
        self.v = v
        self.w = w
        self.x = x
        self.func = func
        self.phi = phi
        self.kphi = kphi
        self.mult = mult

    def __str__(self) -> str:
        return f"{self.u}, {self.v}, {self.w}, {self.x}, {self.func}, {self.phi}, {self.kphi}, {self.mult}"

def _sort_tuple(*args) -> Tuple[Any, Any]:
    num_check, str_check = False, False
    if len(args) == 1:
        args = args[0]
    for u in args:
        num_check = all(isinstance(u, Number) for u in args)
        str_check = all(isinstance(u, str) for u in args)
        if num_check or str_check:
            n = len(args)
            if n == 2:
                return tuple(sorted((args)))
            elif n == 3:
                if args[0] > args[2]:
                    return tuple(args[::-1])
                return tuple(args)
            elif n == 4:
                if args[0] > args[3]:
                    return tuple(args[::-1])
                return tuple(args)
            else:
                raise InternalCoordinateError(f"Expected length was in [2, 4]. {n}) was passed instead.")
