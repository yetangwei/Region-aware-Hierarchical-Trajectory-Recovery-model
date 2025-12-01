import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
import torch.nn.functional as F


class Date2VecConvert:
    def __init__(self, dim, model_path):
        self.model = Date2Vec(k=dim)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.eval()

    def __call__(self, x):
        with torch.no_grad():
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0)# .cpu()


class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(6, k1)

        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)

        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(k // 2, 6)

        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out
    

class Date2vec_M(nn.Module):
    def __init__(self, dim, model_path):
        super(Date2vec_M, self).__init__()
        self.d2v = Date2VecConvert(dim, model_path)

    def forward(self, time_seq):
        one_list = []
        for timestamp in time_seq:
            t = [timestamp.hour, timestamp.minute, timestamp.second, timestamp.year, timestamp.month, timestamp.day]
            x = torch.Tensor(t).float()
            embed = self.d2v(x)
            one_list.append(embed)

        one_list = torch.vstack(one_list).numpy()

        return one_list
    

class add_time_embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mask_time_embedding = nn.Parameter(torch.randn(1, args.hidden_emb_dim))

    def forward(self, time_feat, predict_idx):
        # time_feat: [L, D]  e.g. [41, 16]
        B, L, D = time_feat.shape
        M = len(predict_idx)

        mask_embedding = self.mask_time_embedding.expand(B, M, -1)
        time_feat[:, predict_idx, :] += mask_embedding

        return time_feat
    


class LSTM_gps(nn.Module):
    def __init__(self, args):
        super(LSTM_gps, self).__init__()
        self.args = args
        self.embedding = nn.Linear(2, args.d_model)
        self.lstm = nn.LSTM(input_size=args.d_model,
                            hidden_size=args.d_model,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.norm = nn.LayerNorm(args.d_model)
        self.fc = nn.Linear(args.d_model, args.d_model)

    def forward(self, input_ids):
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L, D]
        x, _ = self.lstm(x)            # [B, L, D]
        x = self.norm(x) 
        logits = self.fc(x)                                             # [B, M, Vocab]
        return logits
    

class LSTM_gps_t(nn.Module):
    def __init__(self, args):
        super(LSTM_gps_t, self).__init__()
        self.args = args
        self.embedding = nn.Linear(2, args.d_model)
        self.lstm = nn.LSTM(input_size=args.d_model,
                            hidden_size=args.d_model,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.norm = nn.LayerNorm(args.d_model)
        self.fc = nn.Linear(args.d_model, args.d_model)

    def forward(self, input_ids):
        # input_ids: [B, L]
        x = input_ids
        # x = self.embedding(input_ids)  # [B, L, D]
        x, _ = self.lstm(x)            # [B, L, D]
        x = self.norm(x) 
        logits = self.fc(x)                                             # [B, M, Vocab]
        return logits
    



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = query

        attn_output, _ = self.attn(query, key, value)
        x = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class ddgam_trans(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key=None, value=None):

        x = query
        for layer in self.layers:
            x = layer(x, key, value)
        return x


def create_2d_sinusoidal_encoding(H, W, d_model=256, temperature=5000.0, device=None, dtype=None):
    assert d_model % 4 == 0
    device = device or torch.device('cpu')
    dtype = dtype or torch.float32
    y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
    x = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)
    dim_t = temperature ** (torch.arange(d_model // 4, device=device, dtype=dtype) * 4.0 / d_model)
    pe = torch.zeros(H, W, d_model, device=device, dtype=dtype)
    pe[..., 0::4] = torch.sin(y / dim_t)[:, None, :]
    pe[..., 1::4] = torch.cos(y / dim_t)[:, None, :]
    pe[..., 2::4] = torch.sin(x / dim_t)[None, :, :]
    pe[..., 3::4] = torch.cos(x / dim_t)[None, :, :]
    return pe

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, H=100, W=100, d_model=256):
        super(LearnablePositionalEncoding, self).__init__()
        initial_pe = create_2d_sinusoidal_encoding(H, W, d_model)
        self.pe = nn.Parameter(initial_pe)
        
    def forward(self):
        return self.pe

def remap_ids_to_scale(ids, G_src, S_tgt):
    s = G_src // S_tgt
    ids = torch.clamp(ids, 0, G_src*G_src-1)
    row = ids // G_src
    col = ids %  G_src
    return (row // s) * S_tgt + (col // s)
    

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class MultiScaleDeformGrid(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.hidden_emb_dim
        self.D = D
        self.conv0 = ConvBNAct(1, D, k=1, s=1, p=0)
        self.deform0 = DeformStage(in_channels=D, out_channels=D, kernel_size=7, offset_scale=4)
        self.conv1_out = torch.nn.Conv2d(
            in_channels=D,
            out_channels=D,  # 2 * kernel_size * kernel_size
            kernel_size=3,
            padding=1
        )

        self.down1 = ConvBNAct(D, D, k=3, s=2)
        self.deform1 = DeformStage(in_channels=D, out_channels=D, kernel_size=5, offset_scale=2)
        self.conv2_out = torch.nn.Conv2d(
            in_channels=D,
            out_channels=D,  # 2 * kernel_size * kernel_size
            kernel_size=3,
            padding=1
        )
    
        self.down2 = ConvBNAct(D, D, k=3, s=2)
        self.deform2 = DeformStage(in_channels=D, out_channels=D, kernel_size=3, offset_scale=1)
        self.conv3_out = torch.nn.Conv2d(
            in_channels=D,
            out_channels=D,  # 2 * kernel_size * kernel_size
            kernel_size=3,
            padding=1
        )



    def _make_pe(self, H, W, device, dtype):
        return create_2d_sinusoidal_encoding(H, W, d_model=self.D, device=device, dtype=dtype)

    def _to_id_embed_l(self, fmap, pe):
        _, D, H, W = fmap.shape
        device, dtype = fmap.device, fmap.dtype
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        x = fmap + pe
        E = x.squeeze(0).permute(1, 2, 0) 
        E = torch.flip(E, dims=[0]) 
        return E.reshape(-1, D), x
    


    def _to_id_embed(self, fmap):
        _, D, H, W = fmap.shape
        device, dtype = fmap.device, fmap.dtype
        pe = self._make_pe(H, W, device, dtype).permute(2, 0, 1).unsqueeze(0)
        x = fmap + pe
        E = x.squeeze(0).permute(1, 2, 0) 
        E = torch.flip(E, dims=[0])     
        return E.reshape(-1, D), x
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_0 = self.conv0(x)
        out_0, offset_0 = self.deform0(x_0)
        # out_0 = self.conv1_out(out_0)

        x_1 = self.down1(out_0 + x_0)
        out_1, offset_1 = self.deform1(x_1)
        # out_1 = self.conv2_out(out_1)

        x_2 = self.down2(out_1 + x_1)
        out_2, offset_2 = self.deform2(x_2)
        # out_2 = self.conv3_out(out_2)

        pe0 = self.position_encoding0()
        pe1 = self.position_encoding1()
        pe2 = self.position_encoding2()


        # E0, out_0 = self._to_id_embed(out_0)
        # E1, out_1 = self._to_id_embed(out_1)
        # E2, out_2 = self._to_id_embed(out_2)

        E0, out_0 = self._to_id_embed_l(out_0, pe0)
        E1, out_1 = self._to_id_embed_l(out_1, pe1)
        E2, out_2 = self._to_id_embed_l(out_2, pe2)

        return E0, out_0, offset_0, E1, out_1, offset_1, E2, out_2, offset_2


class HierDDGAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.G = args.grid_num
        self.D = args.hidden_emb_dim

        # 多尺度编码器
        self.encoder = MultiScaleDeformGrid(args)

        self.grid_mask_token = nn.Parameter(torch.randn(self.D))


        self.delta_proj   = nn.Sequential(nn.Linear(2, self.D//2), nn.ReLU(True), nn.Linear(self.D//2, self.D))
        self.delta_encoder= LSTM_gps_t(args)
        self.gps_encoder  = LSTM_gps(args)
        self.fusion_gps   = nn.Sequential(nn.Linear(self.D*2, self.D), nn.ReLU(True), nn.Linear(self.D, self.D))
        self.fusion_token = nn.Sequential(nn.Linear(self.D*2, self.D), nn.ReLU(True), nn.Linear(self.D, self.D))

        self.seq_transformer = ddgam_trans(2, self.D, args.grid_trm_head, args.grid_trm_dropout)
        # self.transformer = ddgam_trans(args.grid_trm_layer, args.hidden_emb_dim, args.grid_trm_head, args.grid_trm_dropout)


        self.cross_c = ddgam_trans(args.grid_trm_layer, self.D, args.grid_trm_head, args.grid_trm_dropout)
        self.cross_m = ddgam_trans(args.grid_trm_layer, self.D, args.grid_trm_head, args.grid_trm_dropout)
        self.cross_f = ddgam_trans(args.grid_trm_layer, self.D, args.grid_trm_head, args.grid_trm_dropout)

        S_c = self.G // 4; S_m = self.G // 2; S_f = self.G
        self.head_c = nn.Linear(self.D, S_c*S_c)
        self.head_m = nn.Linear(self.D, S_m*S_m)
        self.head_f = nn.Linear(self.D, S_f*S_f)
        self.delta_f = nn.Sequential(nn.Linear(self.D, self.D//2), nn.ReLU(True), nn.Linear(self.D//2, 2), nn.Sigmoid())

        self.q_c_mlp = nn.Sequential(nn.Linear(self.D + self.D, (self.D + self.D)*2), nn.ReLU(True), nn.Linear((self.D + self.D)*2, self.D))
        self.qc_m_mlp = nn.Sequential(nn.Linear(self.D + self.D, (self.D + self.D)*2), nn.ReLU(True), nn.Linear((self.D + self.D)*2, self.D))

    

    def get_pe_1d(self, L, D, device, dtype):
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-math.log(10000.0)/D))
        pe = torch.zeros(L, D, device=device, dtype=dtype)
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        return pe


    def _kv_from_endpoints(self, start_ids, end_ids, out, off, S, K=3, dilation=1):

        B, N = start_ids.shape
        neigh = sample_id_neighborhood(ids, out, off, kernel_size=K)  # [B,K*K,2N,D]
        _, KK, twoN, D = neigh.shape
        kv = neigh.reshape(B, KK * twoN, D)
        return kv
    
    def forward(self, grid_direct, grid_loc, time_embedding, time_embedding_mask, time_embedding_label, \
            grid_seq, grid_seq_mask, grid_seq_label, gps_seq, gps_seq_mask, gps_seq_label, \
            delta_gps_seq, delta_gps_seq_mask, delta_gps_seq_label, gps_seq_non, gps_seq_non_label):
        
        
        G = self.G; S_m = G//2; S_c = G//4
        E0, out_0, offset_0, E1, out_1, offset_1, E2, out_2, offset_2 = self.encoder(grid_loc)
        id_mask_emb = torch.vstack([E0, self.grid_mask_token.to(E0.dtype).unsqueeze(0)])
        mask_id_fine = id_mask_emb.shape[0] - 1
        q_tokens, win_ids, B, N, = self._build_query_tokens(id_mask_emb, grid_seq, grid_seq_mask, gps_seq_mask, delta_gps_seq_mask)
        B, N, L = win_ids.shape

        start_f = torch.where(win_ids[...,0] == mask_id_fine, win_ids[...,1], win_ids[...,0])
        end_f   = torch.where(win_ids[...,-1] == mask_id_fine, win_ids[...,-2], win_ids[...,-1])
        start_m = remap_ids_to_scale(start_f, G, S_m); end_m = remap_ids_to_scale(end_f, G, S_m)
        start_c = remap_ids_to_scale(start_f, G, S_c); end_c = remap_ids_to_scale(end_f, G, S_c)

        kv_c = self._kv_from_endpoints(
            start_c, end_c, out_2, offset_2, S_c,
            K=3, dilation=1
        )
        kv_m = self._kv_from_endpoints(
            start_m, end_m, out_1, offset_1, S_m,
            K=5, dilation=1
        )
        kv_f = self._kv_from_endpoints(
            start_f, end_f, out_0, offset_0, self.G,
            K=7, dilation=1
        )

        feat_c = self.cross_c(q_tokens, kv_c, kv_c)
        q_c = self.q_c_mlp(torch.cat([q_tokens, feat_c], dim=-1))

        feat_m = self.cross_m(q_c, kv_m, kv_m)
        q_cm = self.qc_m_mlp(torch.cat([q_c, feat_m], dim=-1))

        feat_f = self.cross_f(q_cm, kv_f, kv_f)

        logits_c = self.head_c(feat_c)  # [B,Lp,S_c^2]
        logits_m = self.head_m(feat_m)  # [B,Lp,S_m^2]
        logits_f = self.head_f(feat_f)  # [B,Lp,G^2]
        delta_f  = self.delta_f(feat_f) # [B,Lp,2]

        return feat_c, feat_m, feat_f, logits_c, logits_m, logits_f, delta_f










    
