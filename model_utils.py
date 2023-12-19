import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from parsers import args

cuda = torch.cuda.is_available() and not args.no_cuda

class attentive_node_features_dag(nn.Module):
    '''
    Method to obtain attentive node features over the graph convoluted features
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self,features, lengths, nodal_att_type):
        '''
        features : (B, N, V)
        lengths : (B, )
        nodal_att_type : type of the final nodal attention
        '''

        if nodal_att_type==None:
            return features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        padding_mask = [l*[1]+(max_seq_len-l)*[0] for l in lengths]
        padding_mask = torch.tensor(padding_mask).to(features) 
        causal_mask = torch.ones(max_seq_len, max_seq_len).to(features) 
        causal_mask = torch.tril(causal_mask).unsqueeze(0) 

        if nodal_att_type=='global':
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type=='past':
            mask = padding_mask.unsqueeze(1)*causal_mask

        x = self.transform(features) 
        temp = torch.bmm(x, features.permute(0,2,1))
        alpha = F.softmax(torch.tanh(temp), dim=2)
        alpha_masked = alpha*mask
        
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
        alpha = alpha_masked / alpha_sum 
        attn_pool = torch.bmm(alpha, features)

        return attn_pool

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda, max_seq_length):

    start_zero = seq_lengths.data.new(1).zero_()
    
    if not no_cuda:
        start_zero = start_zero.cuda()
    max_len = max_seq_length

    start = torch.cumsum(torch.cat((start_zero, seq_lengths[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                                for s, l in zip(start.data.tolist(),
                                seq_lengths.data.tolist())], 0)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    max_len_ = emotions.size(1)
    for t in range(max_len_):
        att_em, alpha_ = matchatt_layer(emotions, emotions[:,t,:], mask=umask) 
        att_emotions.append(att_em.unsqueeze(1)) 
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=1)
    att_emotions = torch.cat([att_emotions[j, :, :][:seq_lengths[j].item()] for j in range(len(seq_lengths))])

    return att_emotions

def edge_perms(l, window_past, window_future):

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
    
def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(0)

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features
       
def batch_graphify(features, qmask, lengths, window_past, window_future, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification 
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method 
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from 
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces 
    from any other dialogue instances in that mini-batch.
    """
    edge_index, edge_type, node_features = [], [], []
    edge_index_modal = []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []   

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])
        perms1 = edge_perms(lengths[j].item(), window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j].item()
        edge_index_lengths.append(len(perms1))
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)

    edge_index_ =  torch.stack([edge_index[0] + node_features.shape[0], edge_index[1] + node_features.shape[0]],dim=0)
    for i in range(node_features.shape[0]):
        edge_index_modal.append(torch.tensor([i, i+node_features.shape[0]]))
        edge_index_modal.append(torch.tensor([i+node_features.shape[0], i]))
    edge_index_modal_ = torch.stack(edge_index_modal).transpose(0, 1)

    edge_index1 = torch.cat([edge_index,edge_index_,edge_index_modal_], dim=-1)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_index1 = edge_index1.cuda()

    return node_features, edge_index, edge_index_lengths, edge_index1

def batch_graphify_net2(features_a, features_v, features_l):

    node_features = []
    node_features = torch.stack((features_a, features_v, features_l), dim=0)
    adj = torch.ones(len(node_features), len(node_features))
    if cuda:
        node_features = node_features.cuda()
        adj = adj.cuda()

    return node_features, adj

def feature_adj_net2(features_a, features_v, features_l, modals, no_cuda):

        if len(modals) == 3:
            features = torch.cat([features_a, features_v, features_l], dim=0).cuda() if not no_cuda else torch.cat([features_a, features_v, features_l], dim=0)
        elif 'a' in modals and 'v' in modals:
            features = torch.cat([features_a, features_v], dim=0).cuda() if not no_cuda else torch.cat([features_a, features_v], dim=0)
        elif 'a' in modals and 'l' in modals:
            features = torch.cat([features_a, features_l], dim=0).cuda() if not no_cuda else torch.cat([features_a, features_l], dim=0)
        elif 'v' in modals and 'l' in modals:
            features = torch.cat([features_v, features_l], dim=0).cuda() if not no_cuda else torch.cat([features_v, features_l], dim=0)
        else:
            return NotImplementedError
        adj_block = torch.eye(features_a.shape[0]).cuda() if not no_cuda else torch.eye(features_a.shape[0])
        adj = torch.tile(adj_block,(len(modals), len(modals))).cuda() if not no_cuda else torch.tile(adj_block,(len(modals), len(modals)))
        return features, adj

def all_to_batch(features_all, seq_lengths, max_seq_length, no_cuda):
    start_zero = seq_lengths.data.new(1).zero_()
    if not no_cuda:
        start_zero = start_zero.cuda()
    max_len = max_seq_length
    start = torch.cumsum(torch.cat((start_zero, seq_lengths[:-1])).cuda() if not no_cuda else torch.cat((start_zero, seq_lengths[:-1])), 0)
    features_batch = torch.stack([pad(features_all.narrow(0, s, l), max_len, no_cuda)
                                for s, l in zip(start.data.tolist(),
                                seq_lengths.data.tolist())], 0).cuda() if not no_cuda else torch.stack([pad(features_all.narrow(0, s, l), max_len, no_cuda)
                                for s, l in zip(start.data.tolist(), seq_lengths.data.tolist())], 0)
    return features_batch

def batch_to_all(features_batch, seq_lengths, no_cuda):
    features_all = torch.cat([features_batch[j, :, :][:seq_lengths[j].item()] for j in range(len(seq_lengths))]).cuda() if not no_cuda else torch.cat([features_batch[j, :, :][:seq_lengths[j].item()] for j in range(len(seq_lengths))])
    return features_all

class PositionalEncoding(nn.Module): 
    "Implement the PE function." 
    def __init__(self, d_model, dropout, max_len=5000): 
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(p=dropout) 
        pe = torch.zeros(max_len, d_model) 

        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 
    def forward(self, x): 
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return self.dropout(x) 