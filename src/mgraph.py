# adapted from torchdrug.data.Molecule
# https://torchdrug.ai/docs/_modules/torchdrug/data/molecule.html
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


class MolecularGraph:
    
    id2atom = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br']
    id2bond = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
             Chem.rdchem.BondType.TRIPLE] # Chem.rdchem.BondType.AROMATIC

    bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}
    atom2id = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'I': 6, 'Cl': 7, 'Br': 8}
    
    num_edge_type = len(bond2id)
    num_node_type = len(atom2id)

    
    def __init__(self, node_type, edge_index, edge_type):
        self.node_type = node_type
        self.edge_index = edge_index
        self.edge_type = edge_type

        self.num_node = len(self.node_type)
        self.num_edge = len(self.edge_type)
        
        
    def add_node(self, node_type):
        node_type = np.append(self.node_type, node_type)
        return MolecularGraph(node_type, self.edge_index, self.edge_type)
        
        
    def add_edge(self, node_in_index, node_out_index, edge_type):
        new_edge_index = [[node_in_index, node_out_index],
                          [node_out_index, node_in_index]]
        edge_index = np.hstack((self.edge_index, new_edge_index))
        edge_type = np.append(self.edge_type, [edge_type, edge_type])
        return MolecularGraph(self.node_type, edge_index, edge_type)


    @classmethod
    def from_molecule(cls, molecule):
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        
        node_type = cls.get_node_type(molecule)
        edge_list = cls.get_edge_list(molecule)
        
        # dummy in edge_list is to make array of shape (3, n) in case there is no edges
        node_type = np.array(node_type, dtype=np.int64)
        edge_list = np.array(edge_list + [[0, 0, 0]], dtype=np.int64)[:-1].T
        
        return cls(node_type, edge_list[:2], edge_list[2])
    
    
    @classmethod
    def from_smiles(cls, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        return cls.from_molecule(molecule)
    
            
    def to_molecule(self):
        mol = Chem.RWMol()

        for node_id in self.node_type:
            atom_symbol = self.id2atom[node_id.item()]
            mol.AddAtom(Chem.Atom(atom_symbol))

        edge_zip = zip(*self.edge_index[:, ::2], self.edge_type[::2])
        for node1_id, node2_id, edge_type_id in edge_zip:
            mol.AddBond(int(node1_id), int(node2_id), 
                        order=self.id2bond[int(edge_type_id)])

        return mol
    
    
    def unroll(self):
        num_edge = self.num_edge
        num_connections = np.bincount(self.edge_index[0])
        num_edge_removes = 0

        unrolled_graphs = []
        while num_edge_removes < num_edge:

            subgraph = MolecularGraph(
                self.node_type[num_connections > 0],
                self.edge_index[:, :num_edge-num_edge_removes],
                self.edge_type[:num_edge-num_edge_removes],
            )
            unrolled_graphs.append(subgraph)

            num_edge_removes += 2
            remove_node_ids = self.edge_index[:, num_edge-num_edge_removes+1]
            num_connections[remove_node_ids] -= 1

        unrolled_graphs.append(
            MolecularGraph(
                subgraph.node_type[:1],
                subgraph.edge_index[:, :0],
                subgraph.edge_type[:0],
            )
        )
        return list(reversed(unrolled_graphs))
    
        
    @classmethod
    def get_node_type(cls, molecule):
        node_type = []
        for i in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(i)
            node_type.append(cls.atom2id[atom.GetSymbol()])
        return node_type
    
    
    @classmethod
    def get_edge_list(cls, molecule):
        edge_list = []
        for i in range(molecule.GetNumBonds()):
            bond = molecule.GetBondWithIdx(i)
            t = cls.bond2id[str(bond.GetBondType())]
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[u, v, t], [v, u, t]]
        return edge_list
    
    
    @classmethod
    def set_class_attributes(cls, id2atom=None, id2bond=None):
        if id2bond:
            cls.id2bond = id2bond
            cls.bond2id = {str(x): i for i, x in enumerate(id2bond)}
            cls.num_edge_type = len(id2bond)
        if id2atom:
            cls.id2atom = id2atom
            cls.atom2id = {x: i for i, x in enumerate(id2atom)}
            cls.num_node_type = len(id2atom)
    
        
    def __repr__(self):
        return "%s(num_node=%s, num_edge=%s)" \
                    %(type(self).__name__, self.num_node, self.num_edge)
    
    
    


class PackedMolecularGraph:
    
    num_edge_type = MolecularGraph.num_edge_type
    num_node_type = MolecularGraph.num_node_type
    
    def __init__(self, graphs):
        self.graphs = graphs
        self.batch_size = len(graphs)

        self.node_type = torch.from_numpy(np.hstack([g.node_type for g in graphs]))
        self.edge_type = torch.from_numpy(np.hstack([g.edge_type for g in graphs]))
        self.edge_index_per_graph = torch.from_numpy(np.hstack([g.edge_index for g in graphs]))

        self.num_nodes = torch.tensor([g.num_node for g in graphs], dtype=torch.int64)
        self.num_edges = torch.tensor([g.num_edge for g in graphs], dtype=torch.int64)

        self.node2graph = torch.arange(self.batch_size).repeat_interleave(self.num_nodes)
        self.edge2graph = torch.arange(self.batch_size).repeat_interleave(self.num_edges)

        self.num_node = len(self.node_type)
        self.num_edge = len(self.edge_type)

        self.offsets = torch.cumsum(self.num_nodes, dim=0) - self.num_nodes
        self.edge_index = self.edge_index_per_graph + self.offsets.repeat_interleave(self.num_edges)
        self.edge_weight = torch.ones(self.num_edge, dtype=torch.float32)
        
        self.graph_feature = None
        self.node_feature = None
    
    
    @classmethod
    def from_batch_smiles(cls, smiles):
        graphs = [MolecularGraph.from_smiles(smi) for smi in smiles]
        return cls(graphs)
    
    
    @classmethod
    def from_batch_molecule(cls, molecule):
        graphs = [MolecularGraph.from_molecule(mol) for mol in molecule]
        return cls(graphs)
    
    
    def unpack(self):
        return self.graphs
    
    
    def visualize(self, nrows, ncols):
        plt.figure(
            figsize=(
                min(3*ncols, 20), 
                min(3*nrows, 20))
        )
        for i in range(nrows*ncols):
            molecule = self.graphs[i].to_molecule()
            
            plt.subplot(nrows, ncols, i+1)
            plt.title("Molecule %s" % (i+1))
            plt.imshow(Draw.MolToImage(molecule))

            plt.grid("off")
            plt.xticks([])
            plt.yticks([])

        plt.show()

            
        
    def __repr__(self):
        return "%s(batch_size=%s, num_node=%s, num_edge=%s)" \
                    %(type(self).__name__, self.batch_size, self.num_node, self.num_edge)
    
    
    def __len__(self):
        return self.batch_size

    


    



# # adapted from torchdrug.data.Molecule
# # https://torchdrug.ai/docs/_modules/torchdrug/data/molecule.html
# import numpy as np
# from rdkit import Chem
# import matplotlib.pyplot as plt


# class MolecularGraph:
    
#     id2atom = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br']
#     id2bond = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
#              Chem.rdchem.BondType.TRIPLE] # Chem.rdchem.BondType.AROMATIC

#     bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}
#     atom2id = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'I': 6, 'Cl': 7, 'Br': 8}
    
#     num_edge_type = len(bond2id)
#     num_node_type = len(atom2id)
    
#     def __init__(self, node_type, edge_index, edge_type):
#         self.node_type = node_type
#         self.edge_index = edge_index
#         self.edge_type = edge_type

#         self.num_node = self.node_type.size(0)
#         self.num_edge = self.edge_type.size(0)
        
    
#     @classmethod
#     def from_numpy(cls, node_type, edge_index, edge_type):
#         node_type = torch.from_numpy(node_type)
#         edge_index = torch.from_numpy(edge_index)
#         edge_type = torch.from_numpy(edge_type)
#         return cls(node_type, edge_index, edge_type)
            
            
#     @classmethod
#     def from_molecule(cls, molecule):
#         Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        
#         node_type = cls.get_node_type(molecule)
#         edge_list = cls.get_edge_list(molecule)
        
#         # dummy in edge_list is to make array of shape (3, n) in case there is no edges
#         node_type = np.array(node_type, dtype=np.int64)
#         edge_list = np.array(edge_list + [[0, 0, 0]], dtype=np.int64)[:-1].T

        
#         return cls.from_numpy(node_type, edge_list[:2], edge_list[2])
    
    
#     @classmethod
#     def from_smiles(cls, smiles):
#         molecule = Chem.MolFromSmiles(smiles)
#         return cls.from_molecule(molecule)
    
        
#     @classmethod
#     def get_node_type(cls, molecule):
#         node_type = []
#         for i in range(molecule.GetNumAtoms()):
#             atom = molecule.GetAtomWithIdx(i)
#             node_type.append(cls.atom2id[atom.GetSymbol()])
#         return node_type
    
    
#     @classmethod
#     def get_edge_list(cls, molecule):
#         edge_list = []
#         for i in range(molecule.GetNumBonds()):
#             bond = molecule.GetBondWithIdx(i)
#             t = cls.bond2id[str(bond.GetBondType())]
#             u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#             edge_list += [[u, v, t], [v, u, t]]
#         return edge_list
    
    
#     @classmethod
#     def set_class_attributes(cls, id2atom=None, id2bond=None):
#         if id2bond:
#             cls.id2bond = id2bond
#             cls.bond2id = {str(x): i for i, x in enumerate(id2bond)}
#             cls.num_edge_type = len(id2bond)
#         if id2atom:
#             cls.id2atom = id2atom
#             cls.atom2id = {x: i for i, x in enumerate(id2atom)}
#             cls.num_node_type = len(id2atom)
            
            
#     def to_molecule(self):
#         mol = Chem.RWMol()

#         for node_id in self.node_type:
#             atom_symbol = self.id2atom[node_id.item()]
#             mol.AddAtom(Chem.Atom(atom_symbol))

#         edge_zip = zip(*self.edge_index[:, ::2], self.edge_type[::2])
#         for node1_id, node2_id, edge_type_id in edge_zip:
#             mol.AddBond(int(node1_id), int(node2_id), 
#                         order=self.id2bond[int(edge_type_id)])

#         return mol
    
        
#     def __repr__(self):
#         return "%s(num_node=%s, num_edge=%s)" \
#                     %(type(self).__name__, self.num_node, self.num_edge)


# class PackedMolecularGraph(MolecularGraph):
#     def __init__(self, batch_size, node_type, edge_type, edge_index_per_graph, num_nodes, num_edges, node2graph, edge2graph):
        
#         self.batch_size =   batch_size
#         self.node_type =   node_type
#         self.edge_type =  edge_type
#         self.edge_index_per_graph =  edge_index_per_graph
#         self.num_nodes =  num_nodes
#         self.num_edges =  num_edges
#         self.node2graph =  node2graph
#         self.edge2graph = edge2graph

#         self.num_node = len(self.node_type)
#         self.num_edge = len(self.edge_type)
        

#     @classmethod
#     def from_graphs(cls, graphs):
#         batch_size = len(graphs)

#         node_type = torch.hstack([d.node_type for d in graphs])
#         edge_type = torch.hstack([d.edge_type for d in graphs])
#         edge_index_per_graph = torch.hstack([d.edge_index for d in graphs])

#         num_nodes = torch.tensor([d.num_node for d in graphs])
#         num_edges = torch.tensor([d.num_edge for d in graphs])

#         node2graph = torch.arange(batch_size).repeat_interleave(num_nodes)
#         edge2graph = torch.arange(batch_size).repeat_interleave(num_edges)
#         return cls(batch_size, node_type, edge_type, edge_index_per_graph, num_nodes, num_edges, node2graph, edge2graph)


#     @property
#     def edge_index(self):
#         offsets = torch.cumsum(self.num_nodes, dim=0) - self.num_nodes
#         return self.edge_index_per_graph + offsets.repeat_interleave(self.num_edges)


#     @property    
#     def edge_weight(self):
#         return torch.ones(self.num_edge, dtype=torch.float32)


#     @classmethod
#     def from_batch_smiles(cls, smiles):
#         graphs = [MolecularGraph.from_smiles(smi) for smi in smiles]
#         return cls.from_graphs(graphs)
    
    
#     @classmethod
#     def from_batch_molecule(cls, molecule):
#         graphs = [MolecularGraph.from_molecule(mol) for mol in molecule]
#         return cls.from_graphs(graphs)
            
#     def __repr__(self):
#         return "%s(batch_size=%s, num_node=%s, num_edge=%s)" \
#                     %(type(self).__name__, self.batch_size, self.num_node, self.num_edge)
    
#     def __len__(self):
#         return self.batch_size


#     def add_node(self, new_node_type, new_node2graph):
#         assert new_node_type.shape == new_node2graph.shape
#         assert new_node_type.dim() == 1

#         node_type = torch.cat((self.node_type, new_node_type), dim=0)
#         node2graph = torch.cat((self.node2graph, new_node2graph), dim=0)
#         num_nodes = self.num_nodes + torch.bincount(new_node2graph, minlength=self.batch_size)
#         return PackedMolecularGraph(self.batch_size, 
#             node_type, 
#             self.edge_type, 
#             self.edge_index_per_graph, 
#             num_nodes, 
#             self.num_edges, 
#             node2graph, 
#             self.edge2graph,
#         )._node_contiguous()


#     def add_edge(self, node1_index, node2_index, new_edge_type, new_edge2graph):
#         assert new_edge_type.shape == new_edge2graph.shape
#         assert new_edge_type.dim() == 1
        
#         new_edge_index_per_graph = torch.hstack(
#             (
#                 torch.vstack((node1_index, node2_index)),
#                 torch.vstack((node2_index, node1_index))
#             )
#         )
        
#         new_edge_type = new_edge_type.repeat(2)
#         new_edge2graph = new_edge2graph.repeat(2)
        
        
#         edge_index_per_graph = torch.cat((self.edge_index_per_graph, new_edge_index_per_graph), dim=1)
#         edge_type = torch.cat((self.edge_type, new_edge_type), dim=0)
#         edge2graph = torch.cat((self.edge2graph, new_edge2graph), dim=0)
#         num_edges = self.num_edges + torch.bincount(new_edge2graph, minlength=self.batch_size)
#         return PackedMolecularGraph(self.batch_size, 
#                     self.node_type, 
#                     edge_type, 
#                     edge_index_per_graph, 
#                     self.num_nodes, 
#                     num_edges, 
#                     self.node2graph, 
#                     edge2graph
#                 )._edge_contiguous()


#     def _node_contiguous(self):
#         sort_result = torch.sort(self.node2graph, stable=True)
#         self.node2graph = sort_result.values
#         self.node_type = self.node_type[sort_result.indices]
#         return self


#     def _edge_contiguous(self):
#         sort_result = torch.sort(self.edge2graph, stable=True)
#         self.edge2graph = sort_result.values
#         self.edge_type = self.edge_type[sort_result.indices]
#         self.edge_index_per_graph = self.edge_index_per_graph[:, sort_result.indices]
#         return self


#     def boolean_indexing(self, index):
#         assert isinstance(index, torch.Tensor)
#         batch_size = index.sum().item()
#         node_mask = index.repeat_interleave(self.num_nodes)
#         edge_mask = index.repeat_interleave(self.num_edges)

#         # node 
#         node_type = self.node_type[node_mask]
#         num_nodes = self.num_nodes[index]
#         node2graph = torch.arange(batch_size).repeat_interleave(num_nodes)

#         # edge
#         edge_type = self.edge_type[edge_mask]
#         edge_index_per_graph = self.edge_index_per_graph[:, edge_mask]
#         num_edges = self.num_edges[index]
#         edge2graph = torch.arange(batch_size).repeat_interleave(num_edges)

#         return PackedMolecularGraph(batch_size, 
#                             node_type, 
#                             edge_type, 
#                             edge_index_per_graph, 
#                             num_nodes, 
#                             num_edges, 
#                             node2graph, 
#                             edge2graph
#                         )

#     def __getitem__(self, index):
#         assert index < self.batch_size

#         node_mask = self.node2graph == index
#         edge_mask = self.edge2graph == index
#         node_type = self.node_type[node_mask]
#         edge_type = self.edge_type[edge_mask]
#         edge_index = self.edge_index_per_graph[:, edge_mask]
        
#         return MolecularGraph(node_type, edge_index, edge_type)

#     def to_graphs(self):
#         cum_num_nodes = torch.cumsum(self.num_nodes, dim=0)
#         cum_num_edges = torch.cumsum(self.num_edges, dim=0)

#         graphs = []
#         nid0, eid0 = 0, 0
#         for nid1, eid1 in zip(cum_num_nodes, cum_num_edges):
#             node_type = self.node_type[nid0:nid1]
#             edge_type = self.edge_type[eid0:eid1]
#             edge_index = self.edge_index_per_graph[:, eid0:eid1]

#             graphs.append(
#                 MolecularGraph(node_type, edge_index, edge_type)
#             )

#             nid0, eid0 = nid1, eid1
#         return graphs
    
    
#     def visualize(self, nrows, ncols):
#         plt.figure(
#             figsize=(
#                 min(3*ncols, 20), 
#                 min(3*nrows, 20))
#         )
#         for i in range(nrows*ncols):
#             molecule = self.__getitem__(i).to_molecule()
            
#             plt.subplot(nrows, ncols, i+1)
#             plt.title("Molecule %s" % (i+1))
#             plt.imshow(Chem.Draw.MolToImage(molecule))

#             plt.grid("off")
#             plt.xticks([])
#             plt.yticks([])

#         plt.show()
    
    
# # class PairedMolecularData(PackedMolecularGraph):
# #     def __init__(self, graphs, mols):
# #         super(PairedMolecularData, self).__init__(graphs.batch_size, graphs.node_type, graphs.edge_type, graphs.edge_index_per_graph, graphs.num_nodes, graphs.num_edges, graphs.node2graph, graphs.edge2graph)
        
# #         self.graphs = graphs
# #         self.mols = np.array(mols)
    
# #     @classmethod
# #     def from_batch_smiles(cls, smiles):
# #         mols = [Chem.RWMol(Chem.MolFromSmiles(smi)) for smi in smiles]
# #         graphs = PackedMolecularGraph.from_batch_molecule(mols)
# #         return cls(graphs, mols)
    
# #     @classmethod
# #     def from_batch_molecule(cls, molecules):
# #         graphs = PackedMolecularGraph.from_batch_molecule(molecules)
# #         return cls(graphs, molecules)
    
# #     def add_node(self, new_node_type, new_node2graph):
# #         new_graph = self.graphs.add_node(new_node_type, new_node2graph)
        
# #         for gid, nid in zip(new_node2graph, new_node_type):
# #             mol, atom_symbol = self.mols[gid], self.id2atom[nid]
# #             mol.AddAtom(Chem.Atom(atom_symbol))
# #         return PairedMolecularData(new_graph, self.mols)
            
# #     def add_bond(self, node1_index, node2_index, new_edge_type, new_edge2graph):
# #         new_graph = self.graphs.add_edge(node1_index, node2_index, new_edge_type, new_edge2graph)
# #         for n1, n2, eid, gid in zip(node1_index, node2_index, new_edge_type, new_edge2graph):
# #             mol, bond_type = self.mols[gid], self.id2bond[eid]
# #             mol.AddBond(int(n1), int(n2), order=bond_type)
# #         return PairedMolecularData(new_graph, self.mols)
    
# #     def boolean_indexing(self, index):
# #         graphs = super().boolean_indexing(index)
# #         return PairedMolecularData(graphs, self.mols[index])
    
    
# #     def visualize(self, nrows, ncols):
# #         plt.figure(
# #             figsize=(
# #                 min(3*ncols, 20), 
# #                 min(3*nrows, 20))
# #         )
# #         for i in range(nrows*ncols):
# #             molecule = self.mols[i]
            
# #             plt.subplot(nrows, ncols, i+1)
# #             plt.title("Molecule %s" % (i+1))
# #             plt.imshow(Chem.Draw.MolToImage(molecule))

# #             plt.grid("off")
# #             plt.xticks([])
# #             plt.yticks([])

# #         plt.show()
            
# #     def __repr__(self):
# #         return "%s(batch_size=%s, num_node=%s, num_edge=%s)" \
# #                     %(type(self).__name__, self.batch_size, self.num_node, self.num_edge)
    
    

# # class MoleculeData:
# #     id2atom = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br']
# #     id2bond = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
# #              Chem.rdchem.BondType.TRIPLE] # Chem.rdchem.BondType.AROMATIC

# #     bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}
# #     atom2id = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'I': 6, 'Cl': 7, 'Br': 8}
    
# #     num_edge_type = len(bond2id)
# #     num_node_type = len(atom2id)

# #     def __init__(self, node_type, edge_index, edge_type):
# #         self.node_type = node_type
# #         self.edge_index = edge_index
# #         self.edge_type = edge_type

# #         self.num_node = len(self.node_type)
# #         self.num_edge = len(self.edge_type)

# #     def add_node(self, node_type):
# #         self.node_type.append(node_type)
# #         self.num_node = len(self.node_type)

# #     def add_edge(self, node_in_index, node_out_index, edge_type):
# #         self.edge_index[0].extend([node_in_index, node_out_index])
# #         self.edge_index[1].extend([node_out_index, node_in_index])
# #         self.edge_type.extend([edge_type, edge_type])
# #         self.num_edge = len(self.edge_type)

# #     @classmethod
# #     def from_molecule(cls, molecule, kekulize=True):
# #         if kekulize:
# #             Chem.Kekulize(molecule)
        
# #         node_type = cls.get_node_type(molecule)
# #         node_in, node_out, edge_type = cls.get_edge_list(molecule)
# #         return cls(node_type, [node_in, node_out], edge_type)
    
    
# #     @classmethod
# #     def from_smiles(cls, smiles, kekulize=True):
# #         molecule = Chem.MolFromSmiles(smiles)
# #         return cls.from_molecule(molecule, kekulize=kekulize)

    
# #     @classmethod
# #     def get_node_type(cls, molecule):
# #         node_type = []
# #         for i in range(molecule.GetNumAtoms()):
# #             atom = molecule.GetAtomWithIdx(i)
# #             node_type.append(cls.atom2id[atom.GetSymbol()])
# #         return node_type
    
    
# #     @classmethod
# #     def get_edge_list(cls, molecule):
# #         node_in, node_out, edge_type = [], [], []
# #         for i in range(molecule.GetNumBonds()):
# #             bond = molecule.GetBondWithIdx(i)
# #             t = cls.bond2id[str(bond.GetBondType())]
# #             u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
# #             node_in += [u, v]
# #             node_out += [v, u]
# #             edge_type += [t, t]
# #         return node_in, node_out, edge_type
    

# #     @classmethod
# #     def set_class_attributes(cls, id2atom=None, id2bond=None):
# #         if id2bond:
# #             cls.id2bond = id2bond
# #             cls.bond2id = {str(x): i for i, x in enumerate(id2bond)}
# #             cls.num_edge_type = len(id2bond)
# #         if id2atom:
# #             cls.id2atom = id2atom
# #             cls.atom2id = {x: i for i, x in enumerate(id2atom)}
# #             cls.num_node_type = len(id2atom)
            
            
# #     def to_molecule(self):
# #         mol = Chem.RWMol()

# #         for node_id in self.node_type:
# #             atom_symbol = self.id2atom[node_id]
# #             mol.AddAtom(Chem.Atom(atom_symbol))

# #         node_in, node_out = self.edge_index
# #         for i in range(0, self.num_edge, 2):
# #             mol.AddBond(node_in[i], node_out[i], 
# #                         order=self.id2bond[self.edge_type[i]])
# #         return mol
    
        
# #     def __repr__(self):
# #         return "%s(num_node=%s, num_edge=%s)" \
# #                     %(type(self).__name__, self.num_node, self.num_edge)
    
    

# # class MoleculeBatch(MoleculeData):
# #     '''
# #     Additional features in compared to MolecularGraph
# #         :node2graph, edge_weight, num_nodes, num_edges, batch_size, edge_index_per_graph
# #     '''
# #     def __init__(self, data):
# #         self.batch_size = len(data)

# #         self.node_type = np.concatenate([np.array(d.node_type) for d in data])
# #         self.edge_type = np.concatenate([np.array(d.edge_type) for d in data])
# #         self.edge_index_per_graph = np.concatenate([np.array(d.edge_index) for d in data], axis=1)

# #         self.node_type = torch.LongTensor(self.node_type)
# #         self.edge_type = torch.LongTensor(self.edge_type)
# #         self.edge_index_per_graph = torch.LongTensor(self.edge_index_per_graph)
        
# #         self.num_node = len(self.node_type)
# #         self.num_edge = len(self.edge_type)
        
# #         self.num_nodes = torch.tensor([d.num_node for d in data])
# #         self.num_edges = torch.tensor([d.num_edge for d in data])

# #         self.node2graph = torch.arange(self.batch_size).repeat_interleave(self.num_nodes)

# #         start_index = torch.cat((torch.tensor([0]), self.num_nodes[:-1]))
# #         start_index = torch.cumsum(start_index, dim=0)
# #         self.edge_index = self.edge_index_per_graph + start_index.repeat_interleave(self.num_edges)


# #     @property    
# #     def edge_weight(self):
# #         return torch.ones(self.num_edge, dtype=torch.float32)


# #     @classmethod
# #     def from_batch_smiles(cls, smiles, kekulize=True):
# #         graphs = [MoleculeData.from_smiles(smi, kekulize=kekulize) for smi in smiles]
# #         return cls(graphs)
    
    
# #     @classmethod
# #     def from_batch_molecule(cls, molecule, kekulize=True):
# #         graphs = [MoleculeData.from_molecule(mol, kekulize=kekulize) for mol in molecule]
# #         return cls(graphs)
            
# #     def __repr__(self):
# #         return "%s(batch_size=%s, num_node=%s, num_edge=%s)" \
# #                     %(type(self).__name__, self.batch_size, self.num_node, self.num_edge)
