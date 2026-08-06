from __future__ import annotations
from .ff import ForceField, _sort_tuple
from copy import deepcopy
from sys import maxsize
from typing import List, Tuple
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
        self.graph = nx.Graph()
        self.ff = ForceField()

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

    def find_angles(self):
        seqs = []
        blacklist = set()
        for a1 in self.graph.nodes:
            for a2 in self.graph.neighbors(a1):
                for a3 in self.graph.neighbors(a2):
                    if a1 == a3:
                        continue
                    seq = sorted([a1, a3])
                    seq.insert(1, a2)
                    print(seq)
                    seq = tuple(seq)
                    print(seq)
                    if a1 != a3 and seq not in blacklist:
                        seqs.append(seq)
                        blacklist.add(seq)
        seqs.sort()
        self.angles = seqs

    def find_dihedrals(self):
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
                        seq = (a1, a2, a3, a4) if a1 <= a4 else (a4, a3, a2, a1)
                        if a1 != a4 and seq not in blacklist:
                            seqs.append(seq)
                            blacklist.add(seq)
        seqs.sort()
        self.dihedrals = seqs

    def load_itp(self, filepath: str):
        self.ff = ForceField.from_itp(filepath)
    
    def transform(self, x: np.ndarray) -> Molecule:
        new_mol = Molecule()
        return new_mol
    
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

    def dihedrals_energy(self):
        for k in self.ff.dihedrals.keys():
            if set(k) == {'HA3', 'CT3', 'CT1', 'NH3'}:
                print(k)
        print(self.ff.dihedrals)
        e = 0
        for d in self.dihedrals:
            u, v, w, x = d
            type_u = self.atom_types[u]
            type_v = self.atom_types[v]
            type_w = self.atom_types[w]
            type_x = self.atom_types[x]
            print(type_u, type_v, type_w, type_x)
            par = self.ff.dihedrals[_sort_tuple(type_u, type_v, type_w, type_x)]
            phi0 = par.phi
            kphi = par.kphi
            n = par.mult
            p0 = self.coords[u]
            p1 = self.coords[v]
            p2 = self.coords[w]
            p3 = self.coords[x]
            phi = dihedral(p0, p1, p2, p3)
            e += kphi * (1 + np.cons(n * phi - phi0))
        return e

    def impropers_energy(self):
        e = 0
        return e

    def vdw(self):
        e = 0
        return 0

    def elec(self):
        e = 0
        return 0


def distance(p0: np.ndarray, p1: np.ndarray) -> float:
    return np.linalg.norm(p1 - p0)

def urey_bradley_distance():
    pass

def angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return np.dot(p1 - p0, p2 - p0)

def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=1)[:, None] * b1
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))
