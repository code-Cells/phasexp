from __future__ import annotations
from .constants import pi, vacuum_dielectric_constant
from .ff import DihedralParemeter, ForceField, _sort_tuple
from copy import deepcopy
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from sys import maxsize
from typing import Iterable, Iterator, List, Tuple
from itertools import combinations
import networkx as nx
import numpy as np
import periodictable as pt
import re


class Molecule():
    def __init__(self):
        self.atoms = []
        self.atom_types = []
        self.bonds = []
        self.elements = []
        self.coords = []
        self.curr_coords = []
        self.resids = []
        self.resid_atoms = {}
        self.resnames = []
        self.pairs = []
        self.graph = nx.Graph()
        self.ff = ForceField()
        self.vdw = None

    @staticmethod
    def load_gro(gro: str, top: str, ff: str=None) -> Molecule:
        mol = Molecule()
        Molecule._load_topology(mol, top)
        Molecule._load_coordinates(mol, gro)
        Molecule._build_atoms(mol)
        Molecule._build_residues(mol)
        mol.find_angles()
        mol.find_dihedrals()
        if ff:
            mol.ff = ForceField.from_itp(ff)
            # mol.vdw = mol.vdw_1 of mol.ff.comb_rule == 1 else mol.vdw_2
            mol.vdw = mol.vdw_1
        else:
            mol.ff = ForceField()
        return mol

    @staticmethod
    def _load_topology(mol: Molecule, top: str):
        with open(top) as f:
            raw = f.readlines()
        atoms_key = 0
        bonds_key = 0
        pairs_key = 0
        for i, line in enumerate(raw):
            if line.startswith("[ atoms ]"):
                atoms_key = i
            elif line.startswith("[ bonds ]"):
                bonds_key = i
            elif line.startswith("[ pairs ]"):
                pairs_key = i
                break
        for line in raw[atoms_key+3:bonds_key-1]:
            if line.startswith(";"):
                continue
            data = line.split()
            mol.atoms.append(int(data[0])-1)
            mol.atom_types.append(data[1])
        mol.bonds = [tuple(int(i)-1 for i in line.split()[:2]) for line in raw[bonds_key+2:pairs_key-1]]


    @staticmethod
    def _load_coordinates(mol: Molecule, gro: str):
        with open(gro) as f:
            raw = f.readlines()
        curr_res = ""
        for i, line in enumerate(raw[2:-1]):
            data = line.split()
            element_match = re.match(r'[A-Z][a-z]?', data[1])
            element = element_match.group() if element_match else ""
            resid = int(data[0][:-3])
            resname = data[0][3:]
            if curr_res == "":
                mol.resid_atoms[resid] = [i, i]
            elif curr_res != resid:
                if not resid in mol.resid_atoms:
                    mol.resid_atoms[resid] = [i, i]
                mol.resid_atoms[curr_res][1] = i
                mol.resid_atoms[resid][0] = i
            curr_res = resid
            mol.elements.append(element)
            mol.coords.append(np.array([float(j) for j in data[3:]]))
            mol.resids.append(resid)
            mol.resnames.append(resname)
        mol.resid_atoms[curr_res][1] = i+1
        mol.resid_ranges = {
            resid: range(interval[0], interval[1]) 
            for resid, interval in mol.resid_atoms.items()}
        mol.resids = np.asarray(mol.resids)

    @staticmethod
    def _build_atoms(mol: Molecule): 
        mol.graph.add_nodes_from(mol.atoms)
        mol.graph.add_edges_from(mol.bonds)
        keys = "xyz"
        for i in range(len(mol.graph)):
            for j, c in enumerate(mol.coords[i]):
                k = keys[j]
                mol.graph.nodes[i][k] = c
            mol.graph.nodes[i]["element"] = mol.elements[i]

    @staticmethod
    def _build_residues(mol: Molecule):
        resnames = np.asarray(mol.resnames)
        mol.residues = nx.Graph()
        mol.residues.add_nodes_from(np.unique(mol.resids))
        for r, rng in mol.resid_ranges.items():
            n_atoms = np.asarray(list(rng))
            n_resids = np.unique(mol.resids[n_atoms])
            mol.residues.nodes[r]["atoms"] = n_atoms
            mol.residues.nodes[r]["resname"] = resnames[n_atoms[0]]
        for (u, data) in mol.residues.nodes(data=True):
            children = data['atoms']
            for child in children:
                child_neighbors = list(mol.graph.neighbors(child))
                for neighbour in child_neighbors:
                    v = mol.resids[neighbour]
                    if u != v:
                        mol.residues.add_edge(u, v)

    def iter_angles(self):
        blacklist = set()
        for a1 in self.graph.nodes:
            for a2 in self.graph.neighbors(a1):
                for a3 in self.graph.neighbors(a2):
                    if a1 == a3:
                        continue
                    seq = (min(a1, a3), a2, max(a1, a3))
                    if seq not in blacklist:
                        blacklist.add(seq)
                        yield seq  

    def find_angles(self):
        seqs = self.iter_angles() 
        self.angles = sorted(list(seqs))

    def iter_dihedrals(self):
        seqs = []
        blacklist = set()
        for a1 in self.graph.nodes:
            for a2 in self.graph.neighbors(a1):
                for a3 in self.graph.neighbors(a2):
                    if a1 == a3:
                        continue
                    for a4 in self.graph.neighbors(a3):
                        if a2 == a4:
                            continue
                        types = self.atom_types[a1], self.atom_types[a4]
                        if types[0] <= types[1]:
                            seq = (a1, a2, a3, a4)
                        else:
                            seq = (a4, a3, a2, a1)
                        if a1 != a4 and seq not in blacklist:
                            blacklist.add(seq)
                            yield seq

    def find_dihedrals(self):
        seqs = self.iter_dihedrals()
        self.dihedrals = sorted(list(seqs))

    def load_itp(self, filepath: str):
        self.ff = ForceField.from_itp(filepath)
    
    def transform(self, r: NDArray=None, theta: float=None, axis: int | str=None) -> Molecule:
        if (axis is None) != (theta is None):
            raise ValueError("Both axis and theta parameters are necessary to assign rotation") 
        r = np.zeros(3) if r is None else r
        if isinstance(axis, int):
            try:
                axis = "xyz"[axis]
            except:
                raise ValueError("Axis should be 'x', 'y', 'z', 0, 1, or 2")
        new_mol = deepcopy(self)
        rot_mat = Rotation.from_euler(axis, theta, degrees=True)
        new_mol.curr_coords = rot_mat.apply(new_mol.curr_coords)
        new_mol.curr_coords += r
        return new_mol

    def translate(self, vec: NDArray):
        self.curr_coords += vec
    
    def backbone(self) -> nx.Graph:
        bb = deepcopy(self.graph)
        for (node, data) in self.graph.nodes(data=True):
            if data["element"] == "H":
                bb.remove_node(node)
        return bb

    def energy(self) -> float:
        return self.bonds_energy() + self.urey_bradley_energy() + self.angles_energy() + self.dihedrals_energy() + self.impropers_energy() + self.vdw() + self.elec()

    def bonds_energy(self) -> float:
        e = 0
        for b in self.bonds:
            u, v = b
            type_u = self.atom_types[u]
            type_v = self.atom_types[v]
            par = self.ff.bonds[tuple(sorted((type_u, type_v)))]
            b0 = par.b
            kb = par.kb
            p0 = self.coords[u]
            p1 = self.coords[v]
            b = distance(p0, p1)
            e += kb * (b - b0)**2
        return e

    def urey_bradley_energy(self):
        e = 0
        # for a in self.angles:
        #     u, v, w = a.u, a.v, a.w
        #     element_u = self.elements[u]
        #     element_v = self.elements[v]
        #     element_w = self.elements[w]
        #     par = self.ff.angles[(element_u, element_v, element_w)]
        #     ktheta0 = par.ktheta
        #     kub = par.kub
        #     p0 = self.coords[u]
        #     p1 = self.coords[v]
        #     p2 = self.coords[w]
        #     ktheta = urey_bradley_distance(p0, p1, p2)
        #     e += kub * (ktheta - ktheta0)**2
        return e

    def angles_energy(self):
        e = 0
        for a in self.angles:
            u, v, w = a
            type_u = self.atom_types[u]
            type_v = self.atom_types[v]
            type_w = self.atom_types[w]
            par = self.ff.angles[_sort_tuple(type_u, type_v, type_w)]
            theta0 = par.theta
            ub = par.ub
            p0 = self.coords[u]
            p1 = self.coords[v]
            p2 = self.coords[w]
            theta = angle(p0, p1, p2)
            e += ub * (theta - theta0)**2
        return e

    def insert_wildcard(self, types: Iterable[str], pos: int) -> DihedralParemeter:
        for i in range(pos, 4):
            types[i] = "X"
            try:
                par = self.ff.dihedrals[_sort_tuple(types)]
                return par
            except KeyError:
                if pos < 4:
                    par = self.insert_wildcard(types, pos+1)
                    return par

    def dihedrals_energy(self):
        e = 0
        for d in self.dihedrals:
            u, v, w, x = d
            type_u = self.atom_types[u]
            type_v = self.atom_types[v]
            type_w = self.atom_types[w]
            type_x = self.atom_types[x]
            types = [type_u, type_v, type_w, type_x]
            try:
                par = self.ff.dihedrals[_sort_tuple(types)]
            except KeyError:
                par = self.insert_wildcard(types, 0)
            else:
                phi0 = par.phi
                kphi = par.kphi
                n = par.mult
                p0 = self.curr_coords[u]
                p1 = self.curr_coords[v]
                p2 = self.curr_coords[w]
                p3 = self.curr_coords[x]
                phi = dihedral(p0, p1, p2, p3)
                e += kphi * (1 + np.cos(n * phi - phi0))
        return e

    def impropers_energy(self):
        e = 0
        return e

    def vdw_1(self):
        e = 0
        for p in self.pairs:
            try:
                u, v = p
                type_u = self.atom_types[u]
                type_v = self.atom_types[v]
                par = self.ff.pairs[_sort_tuple(type_u, type_v)]
                p0 = self.curr_coords[u]
                p1 = self.curr_coords[v]
                dist = distance(p0, p1) 
                e += par.wii / (4 * dist) - par.vii / (2 * dist)
            except:
                pass
        return e

    def vdw_2(self):
            print(self.atom_types)
            e = 0
            for p in self.pairs:
                try:
                    u, v = p
                    type_u = self.atom_types[u]
                    type_v = self.atom_types[v]
                    par = self.ff.pairs[_sort_tuple(type_u, type_v)]
                    p0 = self.curr_coords[u]
                    p1 = self.curr_coords[v]
                    dist = distance(p0, p1) 
                    e += par.wii / (4 * dist) - par.vii / (2 * dist)
                except:
                    pass
            return e

    def elec(self):
        e = 0
        for p in self.pairs:
            u, v = p
            type_u = self.atom_types[u]
            type_v = self.atom_types[v]
            par_u = self.ff.atom_types[type_u]
            par_v = self.ff.atom_types[type_v]
            p0 = self.curr_coords[u]
            p1 = self.curr_coords[v]
            dist = distance(p0, p1)
            e += par_u.c * par_v.c / (4 * pi * vacuum_dielectric_constant * dist)
        return e

    def get_pairs(self, threshold: Number):
        pairs = []
        for i in self.atoms:
            pi = self.curr_coords[i]
            for j in self.atoms[i+1:]:
                pj = self.curr_coords[j]
                if distance(pi, pj) <= threshold:
                    pairs.append((i, j))
        self.pairs = pairs

    def __str__(self) -> str:
        return NotImplemented

    def __repr__(self) -> str:
        return NotImplemented

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, key: Tuple[slice]) -> int:
        return list(self.iterate(key))

    def iterate(self, key: Tuple[slice]) -> Iterator[int]:
        if not isinstance(key, tuple):
            key = (key,)
        l = len(key)
        if l == 1:
            iterable = self.atoms
        elif l == 2:
            iterable = self.bonds
        elif l == 3:
            iterable = self.angles
        elif l == 4:
            iterable = self.dihedrals
        else:
            raise IndexError(f"Slice index should have 1 to 4 terms")
        for i, pair in enumerate(iterable):
            if self._matches(pair, key):
                yield i
        # if isinstance(key, slice):
        #     return self.atoms[key]
        # atom = self.atoms[key]
        # return atom

    @staticmethod
    def _matches(k, key):
        return all(
            Molecule._matches_one(i, sel)
            for i, sel in zip(k, key))

    @staticmethod
    def _matches_one(i, sel):
        if isinstance(sel, slice):
            return i in range(*sel.indices(i+1))
        return i == sel

    def __iter__(self) -> Iterable[List[int]]:
        return iter(self.atoms)

    def __contains__(self, atom: int) -> bool:
        if atom in self.atoms:
            return True
        return False

    def __eq__(self, other: Molecule) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        match = 0
        match += self.atoms == other.atoms
        match += self.atom_types == other.atom_types
        match += self.bonds == other.bonds
        match += self.elements == other.elements
        match += self.coords == other.coords
        match += self.curr_coords == other.curr_coords
        match += self.resids == other.resids
        match += self.resid_atoms == other.resid_atoms
        match += self.resnames == other.resnames
        match += self.graph == other.graph
        match += self.ff == other.ff
        if match == 11:
            return True
        return False

    def __hash__(self):
        return NotImplemented

    def __matmul__():
        return NotImplemented


def distance(p0: NDArray, p1: NDArray) -> float:
    return np.linalg.norm(p1 - p0)

def urey_bradley_distance():
    pass

def angle(p0: NDArray, p1: NDArray, p2: NDArray) -> float:
    return np.dot(p1 - p0, p2 - p0)

def dihedral(p0: NDArray, p1: NDArray, p2: NDArray, p3: NDArray) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    v1 = np.cross(b0, b1)
    v1 = v1 / (v1 * v1).sum(axis=-1)**0.5
    v2 = np.cross(b1, b2)
    v2 = v2 / (v2 * v2).sum(axis=-1)**0.5
    sig = np.sign((v1 * b2).sum(axis=-1))
    rad = np.arccos((v1*v2).sum(axis=-1) / ((v1**2).sum(axis=-1) * (v2**2).sum(axis=-1))**0.5)
    if sig:
        rad = rad * sig
    return rad

def get_coord(mol: Molecule) -> NDArray:
    return mol.coords

def get_curr_coord(mol: Molecule) -> NDArray:
    return mol.curr_coords
