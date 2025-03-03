import sys

sys.path.append('../')
from model.BasicBert.BertEmbedding import TokenEmbedding
from model.BasicBert.BertEmbedding import PositionalEmbedding
from model.BasicBert.BertEmbedding import SegmentEmbedding
from model.BasicBert.BertEmbedding import BertEmbeddings
from model.BasicBert.BertConfig import BertConfig
import torch

if __name__ == '__main__':
    json_file='../bert-base-chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    src = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    src = src.transpose(0, 1)   # [src_len, batch_size]
    
    # ================= 测试TokenEMbedding =====================
    token_embedding = TokenEmbedding(vocab_size=config.vocab_size, hidden_size=config.hidden_size)
    t_embedding = token_embedding(input_ids=src)
    print('**** ----------------- 测试TokenEMbedding --------------------')
    print('input_token shape [src_len, batch_size]:', src.shape)
    print(f'input_token embedding shape [src_len, batch_size, hidden_size]: {t_embedding.shape}')
    
    # ================= 测试PositionalEmbedding =======================
    position_ids = torch.arange(src.size()[0]).expand((1, -1))
    pos_embedding = PositionalEmbedding(max_position_embeddings=6,
                                        hidden_size=8)
    p_embedding = pos_embedding(position_ids=position_ids)
    # print(pos_embedding.embedding.weight)  # embedding 矩阵
    # print(p_embedding)  # positional embedding 结果,
    print("***** --------- 测试PositionalEmbedding ------------")
    print("position_ids shape [1,src_len]: ", position_ids.shape)
    print(f"pos embedding shape [src_len, 1, hidden_size]: {p_embedding.shape}\n")
