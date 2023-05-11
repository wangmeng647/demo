from torch import nn
from models import Transformer, TransformerConfig
import tools
from trainers import TransformerTrainer, TransformerTrainerConfig

if __name__ == '__main__':
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    transformer_config = TransformerConfig()
    train_iter, src_vocab, tgt_vocab, source, target = tools.load_data_nmt(transformer_config.batch_size,
                                                                           transformer_config.sentence_max_len,
                                                                           file_path='data/fra.txt',
                                                                           num_examples=100
                                                                           )
    transformer_config.vocab_size_encoder = len(src_vocab)
    transformer_config.vocab_size_decoder = len(tgt_vocab)
    model = Transformer(transformer_config)
    model.apply(init_weights)
    trainer_config = TransformerTrainerConfig(epochs=200, device='cpu')
    trainer = TransformerTrainer(trainer_config, model)
    trainer.train(train_iter, tgt_vocab)
