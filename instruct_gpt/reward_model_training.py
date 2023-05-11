import torch
import models
import trainers
from torch.utils import data


if __name__ == '__main__':
    """
    simple example:
    input has the shape torch.size([prompt number, prompt-ans pairs number, sentence_len])
    mask has the same shape with input, prompt and pad is 0
    the second dim of input is prompt-ans pairs, one should rank the pairs so that dim 0 is the head(best one)
    note: the sentence_max_len arg in GPT2Config must equal sentence_len, for the last affine layer
    """
    input_id = torch.tensor([[[1, 2, 0],
                             [2, 3, 4]],

                             [[3, 2, 3],
                             [4, 5, 0]]]
                            )
    mask = torch.tensor([[[0, 1, 0],
                          [0, 0, 1]],

                         [[0, 1, 1],
                          [0, 1, 0]]]
                        )
    gpt2config = models.GPT2Config(sentence_max_len=3, vocab_size=100)
    model = models.GPT2RM(gpt2config)
    trainer_config = trainers.GPT2TrainerConfig()
    trainer = trainers.GPT2RMTrainer(trainer_config, model)
    dataset = data.TensorDataset(input_id, mask)
    data_iter = data.DataLoader(dataset, batch_size=2, shuffle=True)
    trainer.train(data_iter)
