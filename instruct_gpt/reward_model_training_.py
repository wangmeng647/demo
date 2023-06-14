import torch
import models
import trainers
from torch.utils import data
from tokenizer import Tokenizer


if __name__ == '__main__':
    """
    simple example:
    input has the shape torch.size([prompt number, prompt-ans pairs number, sentence_len])
    mask has the same shape with input, prompt and ans is 1, pad is 0
    the second dim of input is prompt-ans pairs, one should rank the pairs so that dim 0 is the head(the best one)
    note: the sentence_max_len arg in GPT2Config must equal sentence_len, for the last affine layer
    """
    input_id = torch.tensor([[[1, 2, 0],
                             [2, 3, 4]],

                             [[3, 2, 3],
                             [4, 5, 0]]]
                            )
    mask = torch.tensor([[[1, 1, 0],
                          [1, 1, 1]],

                         [[1, 1, 1],
                          [1, 1, 0]]]
                        )
    data_train = [['我的的', '的的的啊去'], ['啊我人他', '啊去有额']]
    #data_train = ['我的的', '我的去']
    tokenizer = Tokenizer()
    res = tokenizer.tokenize(data_train)
    tokenizer.build_vocab(res)
    input_id1, mask1 = tokenizer.encode_rank(res)
    input_id1 = torch.tensor(input_id1)
    mask1 = torch.tensor(mask1)
    gpt2config = models.GPT2Config(sentence_max_len=input_id1.shape[-1], vocab_size=100)
    model = models.GPT2RM(gpt2config)
    trainer_config = trainers.GPT2TrainerConfig(device='cpu')
    trainer = trainers.GPT2RMTrainer(trainer_config, model)
    dataset = data.TensorDataset(input_id1, mask1)
    data_iter = data.DataLoader(dataset, batch_size=2, shuffle=True)
    trainer.train(data_iter)
