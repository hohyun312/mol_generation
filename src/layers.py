import torch
from torch import nn
from torch.nn import functional as F
import src.utils as utils

class GatedGraphConv(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_edge_type, batch_norm=False, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edge_type = num_edge_type
        
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = None
            
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.mlp = nn.Linear(num_edge_type * input_dim, hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim,
                                    hidden_size=input_dim,
                                    bias=True)
        

    def message_and_aggregate(self, graph, node_feature):
        assert graph.num_edge_type == self.num_edge_type

        node_in, node_out = graph.edge_index
        node_out = node_out * graph.num_edge_type + graph.edge_type

        adjacency = torch.sparse_coo_tensor(torch.stack([node_out, node_in]), graph.edge_weight,
                                            (graph.num_node * graph.num_edge_type, graph.num_node))
        update = torch.sparse.mm(adjacency, node_feature)
        return update.view(graph.num_node, graph.num_edge_type * self.input_dim)
    

    def combine(self, update, node_feature):
        hidden = self.mlp(update)
        if self.batch_norm:
            hidden = self.batch_norm(hidden)
        hidden = self.activation(hidden)
        output = self.gru(hidden, node_feature)
        return output
    
    
    def forward(self, graph, node_feature):
        update = self.message_and_aggregate(graph, node_feature)
        return self.combine(update, node_feature)
    
    
class MultilayerGatedGraphConv(nn.Module):
    
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 num_edge_type, 
                 batch_norm=False, 
                 activation='relu', 
                 readout="sum"):
                
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.graph_dim, self.node_dim = input_dim, input_dim
        self.num_edge_type = num_edge_type
                        
        self.layers = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.layers.append(GatedGraphConv(self.input_dim, hidden_dim, num_edge_type,
                                              batch_norm, activation))
            
        if readout == "sum":
            self.readout = utils.sum_readout
        elif readout == "mean":
            self.readout = utils.mean_readout
        elif readout == "max":
            self.readout = utils.max_readout
        else:
            raise ValueError("Unknown readout `%s`" % readout)
            
    
    def forward(self, graph, node_feature):
        
        for layer in self.layers:
            node_feature = layer(graph, node_feature)
        
        graph_feature = self.readout(graph, node_feature)

        return graph_feature, node_feature
            

    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, dims, activation='relu'):
        assert len(dims) > 1
        super().__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append( nn.Linear(dims[i-1], dims[i]) )
            
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
            
            
    def forward(self, input):
        output = input
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        return self.layers[-1](output)
    
    
    
class GGNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 feature_dim, 
                 hidden_dims, 
                 num_edge_type=3):
        super().__init__()
        self.mlp = nn.Linear(input_dim, feature_dim)
        self.graph_nn = MultilayerGatedGraphConv(
            feature_dim,
            hidden_dims, 
            num_edge_type,
            batch_norm=False,
            activation="relu",
            readout="sum"
        )
        
        self.input_dim = input_dim
        self.graph_dim, self.node_dim = feature_dim, feature_dim
        
    def forward(self, graph, node_feature):
        hidden = F.relu(self.mlp(node_feature))
        return self.graph_nn(graph, hidden)