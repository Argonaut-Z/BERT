import json
import copy
import six
import logging


class BertConfig(object):
    """Configuration for BertModel"""
    
    def __init__(self,
                 vocab_size=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 pad_token_id=0,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02
                 ):
        """
        Args:
            vocab_size: Vocabulary size of `input_ids`. Defaults to 21128.
            hidden_size: Size of the encoder layers and the pooler layer. Defaults to 768.
            num_hidden_layers: Number of hidden layers in the Transformer encoder. Defaults to 12.
            num_attention_heads: Number of attention heads for each attention layer in the 
                Transformer encoder. Defaults to 12.
            intermediate_size: The size of the "intermediate"(i.e., feed-forward) layer in the 
                Transformer encoder. Defaults to 3072.
            pad_token_id:. Defaults to 0.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. Defaults to 'gelu'.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler. Defaults to 0.1.
            attention_probs_dropout_prob: The dropout ratio for the attention probabilities. Defaults to 0.1.
            max_position_embeddings: The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048). Defaults to 512.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into `BertModel`. Defaults to 2.
            initializer_range:The stdev of the truncated_normal_initializer for initializing all weight matrices. Defaults to 0.02.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        """从json配置文件读取配置信息"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        logging.info(f"成功导入BERT配置文件 {json_file}")
        return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a Json string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

if __name__ == '__main__':
    # 1. 根据json实例化配置对象
    config = BertConfig.from_json_file("../bert_base_chinese/config.json")
    print("加载后的 hidden_size:", config.hidden_size)
    print(config)
    
    # 2. 将配置保存为 JSON 字符串
    json_str = config.to_json_string()
    print("JSON 配置:", json_str)

    # 3. 将配置保存为 Python 字典
    config_dict = config.to_dict()
    print("字典配置:", config_dict)
    
    # 4. 根据Python字典实例化配置
    config1 = BertConfig.from_dict(config_dict)
    print(config1.to_dict())
    
    print(config.__dict__)