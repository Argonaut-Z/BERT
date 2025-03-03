import torch
import torch.nn as nn
from torch.nn.init import normal_


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self._reset_parameters(initializer_range)
    
    def forward(self, input_ids):
        """
        param inputs_ids: [input_ids_len, batch_size]
        return: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)
    
    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class PositionalEmbedding(nn.Module):
    """
    位置编码
        注意： BERT中的位置编码完全不同于Transformer中的位置编码，
              前者本质上也是一个普通的Embedding层，而后者是通过公式计算得到的，
              而这也是为什么BERT只能接受长度为512字符的远呀，因为位置编码的最大size为512：
        Since the position embedding table is a learned variable, we create it using a (long)
        sequence length `max_position_embeddings`. The actual sequence length might be shorter
        than this, for faster training of tasks that do not have long sequences.
                                                                            ————————  Google Research
    """
    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        # 因为BERT预训练模型的长度为512
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameters(initializer_range)
    
    def forward(self, position_ids):
        """
        param position_ids: [1, position_ids_len]
        return: [position_ids_len, 1, hidden_size]
        """
        return self.embedding(position_ids).transpose(0, 1)

    def _reset_paramters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)
    
    def forward(self, token_type_ids):
        """
        param token_type_ids: [token_type_ids_len, batch_size]
        return: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        for p in self.parameters:
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. Token Embedding: normal embedding matrix
        2. Positional Embedding: normal embedding matrix
        3. Segment Embedding: adding sentence segment info, (sent_A:1, sent_B:2)
    """
    
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = TokenEmbedding(vocab_size=config.vocab_size,
                                              hidden_size=config.hidden_size,
                                              initializer_range=config.initializer_range)
        # return: [src_len, batch_size, hidden_size]
        
        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       hidden_size=config.hidden_size,
                                                       initializer_range=config.intializer_range)
        # return: [src_len, batch_size, hidden_size]
        
        self.token_type_embeddings = SegmentEmbedding(type_vocab_size=config.type_vocab_size,
                                                      hidden_size=config.hidden_size,
                                                      initializer_range=config.initializer_range)
        # return: [src_len, batch_size, hidden_size]
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 生成[0, 1, ..., src_len - 1]的张量，再扩充维度为[1, src_len]
        # return: [1, max_position_embeddings]
        
    def forward(self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None):
        """
        param input_ids: 输入序列的原始token id, shape: [src_len, batch_size]
        param position_ids: 位置序列, 本质就是 [0, 1, 2, ..., src_len - 1], shape: [1, src_len]
        param token_type_ids: 句子分隔 token, 例如 [0, 0, 0, 0, 1, 1, 1, 1]用于区分两个句子, shape: [src_len, batch_size]
        return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids)
        # token_embedding.shape = [src_len, batch_size, hidden_size]
        
        if position_ids is None:    # 在实际建模时这个参数其实可以不用传值
            position_ids = self.position_ids
        positional_embedding = self.position_embeddings(position_ids)
        # positional_embedding.shape = [src_len, 1, hidden_size]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids,
                                              device=self.position_ids.device)  # [src_len, batch_size]
        segment_embedding = self.token_type_embeddings(token_type_ids)
        # segment_embedding.shape = [src_len, batch_size, hidden_size]
        
        embeddings = token_embedding + positional_embedding + segment_embedding
        # [src_len, batch_size, hidden_size] + [src_len, 1, hidden_size] + [src_len, batch_size, hidden_size]
        embeddings = self.LayerNorm(embeddings) # [src_len, batch_size, hidden_size]
        embeddings = self.dropout(embeddings)
        return embeddings
    
        
            