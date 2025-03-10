import sys

sys.path.append('../')
from model.BasicBert.MyTransformer import MyMultiheadAttention
from model.BasicBert.MyTransformer import MyTransformerEncoderLayer
from model.BasicBert.MyTransformer import MyTransformerEncoder
from model.BasicBert.MyTransformer import MyTransformerDecoderLayer
from model.BasicBert.MyTransformer import MyTransformerDecoder
from model.BasicBert.MyTransformer import MyTransformer

import torch
import torch.nn as nn

if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_head = 8
    src = torch.rand((src_len, batch_size, d_model))
    src_key_padding_mask = torch.tensor([[False, False, False, True, True],
                                         [False, False, False, False, True]])   # [batch_size, src_len]
    
    tgt = torch.rand((tgt_len, batch_size, d_model))
    tgt_key_padding_mask = torch.tensor([[False, False, False, True, True, True],
                                         [False, False, False, False, True, True]])  # shape: [batch_size, tgt_len]

    # ======================= 测试 MyMultiheadAttention ===========================
    my_mh = MyMultiheadAttention(d_model, num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)
    
    # ======================== 测试 MyTransformerEncoderLayer ========================
    my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model, num_head)
    r = my_transformer_encoder_layer(src=src, src_key_padding_mask=src_key_padding_mask)
    print(r.shape)
    
    # ======================== 测试 MyTransformerEncoder ========================
    my_transformer_encoder = MyTransformerEncoder(encoder_layer=my_transformer_encoder_layer,
                                                  num_layers=2,
                                                  norm=nn.LayerNorm(d_model))
    memory = my_transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)
    print(memory.shape)
    
    # ======================== 测试 MyTransformerDecoderLayer ========================
    my_transformer_decoder_layer = MyTransformerDecoderLayer(d_model, num_head)
    r = my_transformer_decoder_layer(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=src_key_padding_mask)
    print(r.shape)
    
    # ======================== 测试 MyTransformerDecoder ========================
    my_transformer_decoder = MyTransformerDecoder(decoder_layer=my_transformer_decoder_layer,
                                                  num_layers=2,
                                                  norm=nn.LayerNorm(d_model))
    out = my_transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)
    
    # ======================== 测试 MyTransformer ========================
    my_transformer = MyTransformer(d_model=d_model, n_head=num_head, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=500)
    tgt_mask = my_transformer.generate_square_subsequent_mask(tgt_len)
    out = my_transformer(src, tgt=tgt, tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)