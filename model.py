import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dens import DensNet, DensNetLayer
from model_fusion import ConcatFusion, FiLM, GatedFusion, SumFusion
from model_improvedgat import ImprovedGAT, ImprovedGATLayer
from model_mlp import MLP
from model_utils import batch_graphify, simple_batch_graphify

class GNNModel(nn.Module):

    def __init__(self, args, D_m_a, D_m_v, D_m, num_speakers, n_classes):
        
        super(GNNModel, self).__init__()
        self.base_model = args.base_model
        self.no_cuda = args.no_cuda
        self.dropout = args.dropout
        self.modals = [x for x in args.modals]
        self.ratio_modal = args.ratio_modal
        self.multi_modal = args.multi_modal
        self.window_past = args.windowp
        self.window_future = args.windowf
        self.hidesize = args.hidesize
        self.list_mlp = args.list_mlp
        self.ratio_speaker = args.ratio_speaker

        if self.base_model[0] == 'LSTM':
            self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
            self.rnn_audio = nn.LSTM(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[0] == 'GRU':
            self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
            self.rnn_audio = nn.GRU(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[0] == 'Transformer':
            self.linear_audio = nn.Linear(D_m_a, args.hidesize)
            encoder_layer_audio = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
            self.transformer_encoder_audio = nn.TransformerEncoder(encoder_layer_audio, num_layers=args.base_nlayers[0])
        elif self.base_model[0] == 'Dens':
            self.linear_audio = nn.Linear(D_m_a, args.hidesize)
            encoder_layer_audio = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
            self.dens_audio = DensNet(encoder_layer_audio, num_layers=args.base_nlayers[0])
        elif self.base_model[0] == 'None':
            self.linear_audio = nn.Linear(D_m_a, args.hidesize)
        else:
            print ('Base model must be one of .')
            raise NotImplementedError 

        if self.base_model[1] == 'LSTM':
            self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
            self.rnn_visual = nn.LSTM(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[1] == 'GRU':
            self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
            self.rnn_visual = nn.GRU(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[1] == 'Transformer':
            self.linear_visual = nn.Linear(D_m_v, args.hidesize)
            encoder_layer_visual = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
            self.transformer_encoder_visual = nn.TransformerEncoder(encoder_layer_visual, num_layers=args.base_nlayers[1])
        elif self.base_model[1] == 'Dens':
            self.linear_visual = nn.Linear(D_m_v, args.hidesize)
            encoder_layer_visual = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
            self.dens_visual = DensNet(encoder_layer_visual, num_layers=args.base_nlayers[1])
        elif self.base_model[1] == 'None':
            self.linear_visual = nn.Linear(D_m_v, args.hidesize)
        else:
            print ('Base model must be one of .')
            raise NotImplementedError 

        if self.base_model[2] == 'LSTM':
            self.linear_text = nn.Linear(D_m, args.base_size[2])
            self.rnn_text = nn.LSTM(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[2] == 'GRU':
            self.linear_text = nn.Linear(D_m, args.base_size[2])
            self.rnn_text = nn.GRU(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
            self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
        elif self.base_model[2] == 'Transformer':
            self.linear_text = nn.Linear(D_m, args.hidesize)
            encoder_layer_text = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
            self.transformer_encoder_text = nn.TransformerEncoder(encoder_layer_text, num_layers=args.base_nlayers[2])
        elif self.base_model[2] == 'Dens':
            self.linear_text = nn.Linear(D_m, args.hidesize)
            encoder_layer_text = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
            self.dens_text = DensNet(encoder_layer_text, num_layers=args.base_nlayers[2])
        elif self.base_model[2] == 'None':
            self.linear_text = nn.Linear(D_m, args.hidesize)
        else:
            print ('Base model must be one of .')
            raise NotImplementedError 
        if args.ratio_speaker > 0:
            self.speaker_embeddings = nn.Embedding(num_speakers, args.hidesize)

        if args.ratio_modal > 0:
            self.modal_embeddings = nn.Embedding(3, args.hidesize)
        if len(self.modals) ==2:
            if "a" in self.modals and 'v' in self.modals:
                improvedgatlayer_av = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
                self.improvedgat_av = ImprovedGAT(improvedgatlayer_av, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)
            if "a" in self.modals and 'l' in self.modals:
                improvedgatlayer_al = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
                self.improvedgat_al = ImprovedGAT(improvedgatlayer_al, num_layers=args.multimodal_nlayers[1], hidesize=args.hidesize)
            if "v" in self.modals and 'l' in self.modals:
                improvedgatlayer_vl = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
                self.improvedgat_vl = ImprovedGAT(improvedgatlayer_vl, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)    
        if len(self.modals) ==3:
            improvedgatlayer_av = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
            self.improvedgat_av = ImprovedGAT(improvedgatlayer_av, num_layers=args.multimodal_nlayers[0], hidesize=args.hidesize)

            improvedgatlayer_al = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
            self.improvedgat_al = ImprovedGAT(improvedgatlayer_al, num_layers=args.multimodal_nlayers[1], hidesize=args.hidesize)

            improvedgatlayer_vl = ImprovedGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.nheads, use_residual=args.use_residual, no_cuda=args.no_cuda)
            self.improvedgat_vl = ImprovedGAT(improvedgatlayer_vl, num_layers=args.multimodal_nlayers[2], hidesize=args.hidesize)

        if args.fusion_method == 'sum':
            self.fusion_avl = SumFusion(input_dim=args.hidesize, output_dim=args.hidesize)
        elif args.fusion_method == 'concat':
            if len(self.modals)==2:
                self.fusion_avl = ConcatFusion(len(self.modals), input_dim=2*args.hidesize, output_dim=args.hidesize)
            if len(self.modals)==3:
                self.fusion_avl = ConcatFusion(len(self.modals), input_dim=3*args.hidesize, output_dim=args.hidesize)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(args.fusion_method))
        
        if args.list_mlp != []:
            self.mlp = MLP(args.hidesize, args.list_mlp, args.dropout, activation='gelu')
            self.smax_fc = nn.Linear(args.list_mlp[-1], n_classes)
        else:
            self.smax_fc = nn.Linear(args.hidesize, n_classes)

    def forward(self, U, qmask, umask, seq_lengths, max_seq_length, U_a=None, U_v=None):
        if self.base_model[0] == 'LSTM':
            U_a = self.linear_audio(U_a)
            U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_audio.flatten_parameters()
            emotions_a, hidden_a = self.rnn_audio(U_a)
            emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True)
            emotions_a = self.linear_audio_(emotions_a)
        elif self.base_model[0] == 'GRU':
            U_a = self.linear_audio(U_a)
            U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_audio.flatten_parameters()
            emotions_a, hidden_a = self.rnn_audio(U_a)
            emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True)
            emotions_a = self.linear_audio_(emotions_a)
        elif self.base_model[0] == 'Transformer':
            U_a = self.linear_audio(U_a)
            emotions_a = self.transformer_encoder_audio(U_a, src_key_padding_mask=umask)
        elif self.base_model[0] == 'Dens':
            U_a = self.linear_audio(U_a)
            emotions_a = self.dens_audio(U_a)
        elif self.base_model[0] == 'None':
            emotions_a = torch.tanh(self.linear_audio(U_a))

        if self.base_model[1] == 'LSTM':
            U_v = self.linear_visual(U_v)
            U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_visual.flatten_parameters()
            emotions_v, hidden_v = self.rnn_visual(U_v)
            emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
            emotions_v = self.linear_visual_(emotions_v)
        elif self.base_model[1] == 'GRU':
            U_v = self.linear_visual(U_v)
            U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_visual.flatten_parameters()
            emotions_v, hidden_v = self.rnn_visual(U_v)
            emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
            emotions_v = self.linear_visual_(emotions_v)
        elif self.base_model[1] == 'Transformer':
            U_v = self.linear_visual(U_v)
            emotions_v = self.transformer_encoder_visual(U_v, src_key_padding_mask=umask)
        elif self.base_model[1] == 'Dens':
            U_v = self.linear_visual(U_v)
            emotions_v = self.dens_visual(U_v)
        elif self.base_model[1] == 'None':
            emotions_v = torch.tanh(self.linear_visual(U_v))

        if self.base_model[2] == 'LSTM':
            U = self.linear_text(U)
            U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_text.flatten_parameters()
            emotions_l, hidden_l = self.rnn_text(U)
            emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True)
            emotions_l = self.linear_text_(emotions_l)
        elif self.base_model[2] == 'GRU':
            U = self.linear_text(U)
            U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.rnn_text.flatten_parameters()
            emotions_l, hidden_l = self.rnn_text(U)
            emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True)
            emotions_l = self.linear_text_(emotions_l)
        elif self.base_model[2] == 'Transformer':
            U = self.linear_text(U)
            emotions_l = self.transformer_encoder_text(U, src_key_padding_mask=umask)
        elif self.base_model[2] == 'Dens':
            U = self.linear_text(U)
            emotions_l = self.dens_text(U)
        elif self.base_model[2] == 'None':
            emotions_l = torch.tanh(self.linear_text(U))
        if len(self.modals) ==2:
            if "a" in self.modals and 'v' in self.modals:
                features_a, edge_index, edge_index_lengths, edge_index1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.no_cuda)
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            if "a" in self.modals and 'l' in self.modals:
                features_a, edge_index, edge_index_lengths, edge_index1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.no_cuda)
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            if "v" in self.modals and 'l' in self.modals:
                features_v, edge_index, edge_index_lengths, edge_index1 = batch_graphify(emotions_v, qmask, seq_lengths, self.window_past, self.window_future, self.no_cuda)
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        if len(self.modals) ==3:
            features_a, edge_index, edge_index_lengths, edge_index1 = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.no_cuda)
            features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)

        if self.ratio_modal > 0:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)
            features_a = features_a + self.ratio_modal*emb_vector[0].reshape(1, -1).expand(features_a.shape[0], features_a.shape[1])
            features_v = features_v + self.ratio_modal*emb_vector[1].reshape(1, -1).expand(features_v.shape[0], features_v.shape[1])
            features_l = features_l + self.ratio_modal*emb_vector[2].reshape(1, -1).expand(features_l.shape[0], features_l.shape[1])
        if self.ratio_speaker > 0:
            qmask_ = torch.cat([qmask[i,:x,:] for i,x in enumerate(seq_lengths)],dim=0)
            spk_idx = torch.argmax(qmask_, dim=-1).cuda() if not self.no_cuda else torch.argmax(qmask_, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            if len(self.modals) ==2:
                if "a" in self.modals and 'v' in self.modals:
                    features_a = features_a + self.ratio_speaker*spk_emb_vector
                    features_v = features_v + self.ratio_speaker*spk_emb_vector
                if "a" in self.modals and 'l' in self.modals:
                    features_a = features_a + self.ratio_speaker*spk_emb_vector
                    features_l = features_l + self.ratio_speaker*spk_emb_vector
                if "v" in self.modals and 'l' in self.modals:    
                    features_v = features_v + self.ratio_speaker*spk_emb_vector
                    features_l = features_l + self.ratio_speaker*spk_emb_vector
            if len(self.modals) ==3:
                features_a = features_a + self.ratio_speaker*spk_emb_vector
                features_v = features_v + self.ratio_speaker*spk_emb_vector
                features_l = features_l + self.ratio_speaker*spk_emb_vector
        if len(self.modals) ==2:
            if "a" in self.modals and 'v' in self.modals:
                features_single_av = torch.cat([features_a, features_v], dim=0)
                features_cross_av = self.improvedgat_av(features_single_av, edge_index1)
                features_cross_a0, features_cross_v0 = torch.chunk(features_cross_av, 2, dim=0)

                features_avl = self.fusion_avl(features_cross_a0, features_cross_v0, features_cross_v0)
            if "a" in self.modals and 'l' in self.modals:
                features_single_al = torch.cat([features_a, features_l], dim=0)
                features_cross_al = self.improvedgat_al(features_single_al, edge_index1)
                features_cross_a1, features_cross_l0 = torch.chunk(features_cross_al, 2, dim=0)

                features_avl = self.fusion_avl(features_cross_a1, features_cross_l0, features_cross_l0)
            if "v" in self.modals and 'l' in self.modals:    
                features_single_vl = torch.cat([features_v, features_l], dim=0)
                features_cross_vl = self.improvedgat_vl(features_single_vl, edge_index1)
                features_cross_v1, features_cross_l1 = torch.chunk(features_cross_vl, 2, dim=0)

                features_avl = self.fusion_avl(features_cross_v1, features_cross_l1, features_cross_l1)
        if len(self.modals) ==3:
            features_single_av = torch.cat([features_a, features_v], dim=0)
            features_cross_av = self.improvedgat_av(features_single_av, edge_index1)
            features_cross_a0, features_cross_v0 = torch.chunk(features_cross_av, 2, dim=0)

            features_single_al = torch.cat([features_a, features_l], dim=0)
            features_cross_al = self.improvedgat_al(features_single_al, edge_index1)
            features_cross_a1, features_cross_l0 = torch.chunk(features_cross_al, 2, dim=0)

            features_single_vl = torch.cat([features_v, features_l], dim=0)
            features_cross_vl = self.improvedgat_vl(features_single_vl, edge_index1)
            features_cross_v1, features_cross_l1 = torch.chunk(features_cross_vl, 2, dim=0)

            features_cross_a = features_cross_a0 + features_cross_a1
            features_cross_v = features_cross_v0 + features_cross_v1
            features_cross_l = features_cross_l0 + features_cross_l1

            features_avl = self.fusion_avl(features_cross_a, features_cross_v, features_cross_l)

        if self.list_mlp != []:
            prob = self.smax_fc(self.mlp(features_avl))

        else:
            prob = self.smax_fc(features_avl)

        return prob
