import random
from rdkit import Chem

from src.mgraph import MolecularGraph
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False
    
    
class AtomReward:
    def __init__(self, symbol):
        self.atom_id = MolecularGraph.atom2id[symbol]
        self.atom2valence = {'C' : 4,
            "N" : 3,
            'O' : 2,
            'S' : 6,
            'P' : 5,
            'F' : 1,
            'I' : 7,
            'Cl' : 1,
            'Br' : 1
        }

    def get_reward(self, old_graphs, actions, new_graphs):
        rewards = []
        for old, new, action in zip(old_graphs, new_graphs, actions):
            valency_rew, atom_type_rew, num_atom_rew = 0, 0, 0

            node1_id = action[0]
            node2_id = min(old.num_node, action[1])
            for node_id in (node1_id, node2_id):
                node_tp = new.node_type[node_id]
                node_val = np.sum(new.edge_type[new.edge_index[0] == node_id] + 1)

                if node_val > self.atom2valence[MolecularGraph.id2atom[node_tp]]:
                    valency_rew = -1

            if action[3]:
                atom_type_rew = np.sum(new.node_type == self.atom_id) # N:1, O:2, F:5
                num_atom_rew = - max(15 - new.num_node, new.num_node - 25) # num atoms within 15-25

            rewards.append(valency_rew + atom_type_rew + num_atom_rew)
        return np.array(rewards, dtype=np.float32)

    

class GCPNMoleculeEnv:
    # adapted from GCPN github
    # https://github.com/bowenliu16/rl_graph_generation/blob/master/gym-molecule/gym_molecule/envs/molecule.py
        
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        
    def __init__(self, 
                 n_envs=1,
                 initial_states=["C"],
                 max_atom=38,
                 atom_types=['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br'],
                 reward_fn=None
                 ):
        if reward_fn is None:
            self.reward_fn = lambda old_graphs, actions, new_graphs: np.zeros(len(old_graphs), dtype=np.float32)
        else:
            self.reward_fn = reward_fn
        self.n_envs = n_envs
        self.initial_states = initial_states
        self.max_atom = max_atom
        self.atom_types = atom_types
        self.mols = None
        self.graphs = None
        
        
    def reset(self):
        conditional = random.choices(self.initial_states, k=self.n_envs)
        self.mols = []
        for c in conditional:
            mol = Chem.RWMol(Chem.MolFromSmiles(c))
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            self.mols.append(mol)
        
        self.graphs = [MolecularGraph.from_molecule(mol) for mol in self.mols]
        return self.graphs
    
    
    def _get_initial_mol(self):
        conditional = random.sample(self.initial_states, k=1)[0]
        mol = Chem.RWMol(Chem.MolFromSmiles(conditional))
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol
    
        
    def step(self, actions):
        node1_index, node2_index, edge_type, stops = actions.T
        new_mols = self.mol_step(node1_index, node2_index, edge_type)
        new_graphs = self.graph_step(node1_index, node2_index, edge_type)
        
        info = [{} for _ in range(self.n_envs)]
        
        
        num_nodes = np.array([g.num_node for g in new_graphs])
        dones = stops | (num_nodes >= self.max_atom)
        
        rewards = self.reward_fn(self.graphs, actions, new_graphs)
        
        for i, done in enumerate(dones):
            if done:
                info[i]['terminal_observation'] = new_graphs[i]
                info[i]['terminal_mol'] = new_mols[i]
                mol = self._get_initial_mol()
                new_mols[i] = mol
                new_graphs[i] = MolecularGraph.from_molecule(mol)
        
        
        self.mols = new_mols
        self.graphs = new_graphs

        new_graphs = np.array(new_graphs, dtype=object)
        dones = dones.astype(bool)
        info = np.array(info, dtype=object)

        return new_graphs, rewards, dones, info
    
    
    def graph_step(self, node1_index, node2_index, edge_type):
        new_graphs = []
        for i, graph in enumerate(self.graphs):
            node2_index_i = node2_index[i]
            node_type = node2_index_i - graph.num_node
            if node_type >= 0:
                node2_index_i = graph.num_node
                graph = graph.add_node(node_type)
            graph = graph.add_edge(node1_index[i], node2_index_i, edge_type[i])
            new_graphs.append(graph)
        return new_graphs
    
    
    def mol_step(self, node1_index, node2_index, edge_type):
        new_mols = []
        for i in range(self.n_envs):
            new_mol = self._add_bond(self.mols[i], int(node1_index[i]), 
                                 int(node2_index[i]), int(edge_type[i]))
            new_mols.append(new_mol)
        return new_mols
    
    
    def _add_bond(self, mol, first_node_id, second_node_id, bond_type_id):
        if mol.GetNumAtoms() <= second_node_id:
            atom_symbol = self.atom_types[second_node_id - mol.GetNumAtoms()]
            second_node_id = mol.AddAtom(Chem.Atom(atom_symbol))
        mol.AddBond(first_node_id, second_node_id, order=self.bond_types[bond_type_id])
        return mol
    
    
    def set_reward_fn(self, reward_fn):
        self.reward_fn = reward_fn
