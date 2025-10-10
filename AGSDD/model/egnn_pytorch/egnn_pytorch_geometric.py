import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# types
from torch_geometric.nn.dense.linear import Linear
from typing import Optional, List, Union

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    
    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

from .egnn_pytorch import *
import math
from torch_geometric.utils import softmax
# global linear attention
import torch_geometric.utils as utils

class Attention_Sparse(Attention):
    def __init__(self,  dim, heads = 8, dim_head = 64):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(dim, heads = 8, dim_head = 64)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None: 
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1: 
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze() # get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi,n_idxs in zip(*batch_uniques):
                x_list.append( 
                    self.sparse_forward(
                        x[aux_count:aux_count+n_idxs], 
                        context[aux_count:aux_count+n_idxs],
                        batch_uniques = (bi.unsqueeze(-1), n_idxs.unsqueeze(-1)) 
                    ) 
                )
            return torch.cat(x_list, dim=0)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = torch_geometric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geometric.nn.norm.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim, heads, dim_head)
        self.attn2 = Attention_Sparse(dim, heads, dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geometric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask = mask)
        out     = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x =  out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, in_channel, emb_dim, att_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.in_channel = in_channel
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.dropout = nn.Dropout(self.att_dropout)
        self.norm = nn.ModuleList(torch.nn.LayerNorm(self.in_channel) for _ in range(2))

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(self.in_channel, emb_dim, bias=False)
        self.Wk = nn.Linear(self.in_channel, emb_dim, bias=False)
        self.Wv = nn.Linear(self.in_channel, emb_dim, bias=False)
        self.wo = nn.Linear(self.emb_dim, self.in_channel, bias=False)  # add.

        self.fc = nn.Sequential(nn.Linear(self.in_channel, self.in_channel*2), 
                                self.dropout,  # add.
                                nn.SiLU(),
                                nn.Linear(self.in_channel*2, self.in_channel))

    def forward(self, x,keys, pad_mask=None):

        batch_size = x.size(0)
        Q = self.Wq(x)
        K = self.Wk(keys)
        V = self.Wv(keys)

        
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1)
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill((~pad_mask), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
    
        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        output = torch.matmul(att_weights, V)
    
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
        output = self.wo(output)
        inter_var = self.norm[0](output+x)       # add post norm.
        output = self.norm[1](self.fc(inter_var)+inter_var)
        return output 


class Type_Attn(nn.Module):
    def __init__(self, feats_dim) -> None:
        super(Type_Attn, self).__init__()
        self.feats_dim = feats_dim
        self.dropout = nn.Dropout(0.1)
        self.lin_query = nn.Linear(feats_dim, feats_dim, bias=False)
        self.lin_key = nn.Linear(feats_dim, feats_dim, bias=False)
        self.lin_value = nn.Linear(feats_dim, feats_dim, bias=False)
        self.norm2 = torch.nn.LayerNorm(self.feats_dim)
        self.norm3 = torch.nn.LayerNorm(self.feats_dim)
    
        self.to_feats = nn.Linear(feats_dim, feats_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(feats_dim, feats_dim*2),
            self.dropout,
            SiLU(),
            nn.Linear(self.feats_dim*2, self.feats_dim)
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, labels, hiddens, gt, num, saved_dirs=None):
        '''
        gt: [N,20]
        '''
        q = self.lin_query(hiddens)
        k, v = self.lin_key(labels), self.lin_value(labels)
        label_score = q@k.T / (self.feats_dim**0.5)  # [N, 20]
        attn_loss = self.ce(label_score, gt.float())

        label_score_drop = F.dropout(F.softmax(label_score, dim=-1), 0.1)
        node_weight_labels = torch.matmul(label_score_drop, v)
        feats = self.to_feats(node_weight_labels)
        res = self.norm2(hiddens + feats)
        return self.norm3(self.ffn(res)+res), attn_loss


# define pytorch-geometric equivalents
class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        soft_edge = 0,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_edge = False,
        update_coors = False, 
        dropout = 0.,
        coor_weights_clamp_value = None, 
        aggr = "add",
        mlp_num = 2,
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.update_edge = update_edge
        self.coor_weights_clamp_value = None
        self.mlp_num = mlp_num
        self.edge_dim = edge_attr_dim + feats_dim * 2
        self.edge_input_dim = edge_attr_dim
        self.message_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        if self.mlp_num >2:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_input_dim, self.edge_input_dim * 8),
                self.dropout,
                SiLU(),
                nn.Linear(self.edge_input_dim * 8, self.edge_input_dim * 4),
                self.dropout,
                SiLU(),
                nn.Linear(self.edge_input_dim * 4, self.edge_input_dim * 2),
                self.dropout,
                SiLU(),
                nn.Linear(self.edge_input_dim * 2, m_dim),
                SiLU(),
            ) if update_feats else None            
        else:        
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_dim, self.edge_input_dim * 2), 
                self.dropout,
                SiLU(),
                nn.Linear(self.edge_input_dim * 2, self.edge_input_dim),
                SiLU()
            )
        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                         nn.Sigmoid()
        ) if soft_edge else None

        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.edge_norm = torch_geometric.nn.norm.LayerNorm(self.edge_input_dim) if self.update_edge  else None
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()
        if self.mlp_num >2:
            self.node_mlp = nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 8),
                self.dropout,
                SiLU(),
                nn.Linear(feats_dim * 8, feats_dim * 4),
                self.dropout,
                SiLU(),
                nn.Linear(feats_dim * 4, feats_dim * 2),
                self.dropout,
                SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            ) if update_feats else None            
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2), 
                self.dropout,
                SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            ) if update_feats else None

        self.type_attn = Type_Attn(feats_dim=feats_dim)

        self.gru = nn.GRUCell(input_size=self.edge_input_dim, hidden_size=feats_dim*2)

        self.hidden_mlp = nn.Sequential(
            nn.Linear(feats_dim*2, feats_dim*2),
            self.dropout,
            SiLU(),   # distance must > 0.
            nn.Linear(feats_dim*2, feats_dim),
            SiLU()
        )

        self.cls_emb = nn.Embedding(1, self.feats_dim)
        self.MHA = MultiHeadAttention(8, in_channel=self.feats_dim, emb_dim=64)
        self.gru_attn = nn.GRUCell(input_size=self.feats_dim, hidden_size=self.feats_dim)

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:

            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None, labels_embed=None, gt=None, layer_num=None, saved_dirs=None, pdb_id=None,
                angle_data: List = None,  size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (2, n_edges)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)  # [N,1]

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d') # d= 2*fourier_features+1

        hidden_out, coors_out, attn_loss = self.propagate(edge_index,x=feats, edge_attr=edge_attr,
                                               coors=coors, rel_coors=rel_coors, rel_dist=rel_dist,batch=batch, labels=labels_embed, gt=gt, layer_num=layer_num, saved_dirs=saved_dirs)
        if self.update_edge:
            # post
            edge_batch = batch[edge_index[0]] 
            edge_attr_feats = self.edge_mlp(torch.cat([hidden_out[edge_index[0]], edge_attr, hidden_out[edge_index[1]]], -1))
            edge_attr = self.edge_norm(self.dropout(edge_attr_feats) + edge_attr,edge_batch)
            #######
            return torch.cat([coors_out, hidden_out], dim=-1),edge_attr, attn_loss
        else:
            return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr, rel_dist) -> Tensor:  # rel_dist
        hiddens = torch.cat([x_i, x_j], dim=-1)
        m_ij = self.gru(edge_attr, hiddens)
        m_ij = self.hidden_mlp(m_ij)
        return m_ij, None

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        try:
            size = self.__check_input__(edge_index, size)
            coll_dict = self.__collect__(self.__user_args__,
                                        edge_index, size, kwargs)
        except AttributeError:
            size = self._check_input(edge_index, size)
            coll_dict = self._collect(self.__user_args__,
                                        edge_index, size, kwargs) 
                       
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        # get messages
        m_ij, diss = self.message(**msg_kwargs)

        # update coors if speified
        coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim = -1) )  # using GNN to update h_i.

            hidden_out = self.node_norm(hidden_out, kwargs["batch"])

            cls = torch.arange(1).cuda()
            cls_embed = self.cls_emb(cls)  
            cls_embed = cls_embed.repeat(torch.unique(kwargs["batch"]).shape[0],1).unsqueeze(1) 
          
            bs_node, mask = utils.to_dense_batch(hidden_out, kwargs["batch"])
            update_cls = self.MHA(cls_embed, bs_node, mask)  
            to_sparse_cls = update_cls.repeat(1, mask.shape[1], 1)[mask]  

            from_cls_feats = self.gru_attn(hidden_out, to_sparse_cls)
            
            res_from_cls_feats = from_cls_feats + hidden_out # residual connection.
            
            res_from_cls_feats = self.node_norm(res_from_cls_feats, kwargs["batch"]) 
            
            node_weight_label, attn_loss = self.type_attn(kwargs['labels'], res_from_cls_feats, kwargs['gt'], kwargs['layer_num'], kwargs['saved_dirs']) 

            out = kwargs["x"] + node_weight_label
            


        else: 
            out = kwargs["x"]

  
        return self.update((out, coors_out, attn_loss), **update_kwargs) 

    def __repr__(self):
        dict_print = {}
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__) 


class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """
    def __init__(self, n_layers, feats_dim, 
                 pos_dim = 3,
                 edge_attr_dim = 0, 
                 m_dim = 16,
                 fourier_features = 0, 
                 soft_edge = 0,
                 embedding_nums=[], 
                 embedding_dims=[],
                 edge_embedding_nums=[], 
                 edge_embedding_dims=[],
                 update_coors=True, 
                 update_feats=True, 
                 norm_feats=True, 
                 norm_coors=False,
                 norm_coors_scale_init = 1e-2, 
                 dropout=0.,
                 coor_weights_clamp_value=None, 
                 aggr="add",
                 global_linear_attn_every = 0,
                 global_linear_attn_heads = 8,
                 global_linear_attn_dim_head = 64,
                 num_global_tokens = 4,
                 recalc=0 ,):
        super().__init__()

        self.n_layers         = n_layers 

        # Embeddings? solve here
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers      = nn.ModuleList()
        self.feats_dim        = feats_dim
        self.pos_dim          = pos_dim
        self.edge_attr_dim    = edge_attr_dim
        self.m_dim            = m_dim
        self.fourier_features = fourier_features
        self.soft_edge        = soft_edge
        self.norm_feats       = norm_feats
        self.norm_coors       = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats     = update_feats
        self.update_coors     = update_coors
        self.dropout          = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc           = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, self.feats_dim))
        
        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_Sparse(feats_dim = feats_dim,
                                pos_dim = pos_dim,
                                edge_attr_dim = edge_attr_dim,
                                m_dim = m_dim,
                                fourier_features = fourier_features, 
                                soft_edge = soft_edge, 
                                norm_feats = norm_feats,
                                norm_coors = norm_coors,
                                norm_coors_scale_init = norm_coors_scale_init, 
                                update_feats = update_feats,
                                update_coors = update_coors, 
                                dropout = dropout, 
                                coor_weights_clamp_value = coor_weights_clamp_value)

            # global attention case
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention_Sparse(dim = self.feats_dim, 
                                                   heads = global_linear_attn_heads, 
                                                   dim_head = global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([attn_layer,layer]))
            # normal case
            else: 
                self.mpnn_layers.append(layer)
            

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        # regulates wether to embedd edges each layer
        edges_need_embedding = False  
        for i,layer in enumerate(self.mpnn_layers):
            
            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            # attn tokens
            self.global_tokens = None
            if exists(self.global_tokens):
                unique, amounts = torch.unique(batch, return_counts=True)
                num_idxs = torch.cat([torch.arange(num_idxs_i,device=self.global_tokens.device) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            # pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else: 
                # only pass feats to the attn layer
                # unique, amounts = torch.unique(batch, return_counts=True)
                x_attn = layer[0](x[:, self.pos_dim:], x[:, self.pos_dim:],batch)[0]#global_tokens
                # merge attn-ed feats and coords
                x = torch.cat( (x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i%self.recalc == 0) and not (i == len(self.mpnn_layers)-1)) :
                edge_index, edge_attr, _ = recalc_edge(x) # returns attr, idx, any_other_info
                edges_need_embedding = True
            
        return x

    def __repr__(self):
        return 'EGNN_Sparse_Network of: {0} layers'.format(len(self.mpnn_layers))