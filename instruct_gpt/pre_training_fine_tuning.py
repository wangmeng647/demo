import torch
import models
import trainers
from torch.utils import data


if __name__ == '__main__':

    gpt2config = models.GPT2Config(vocab_size=100)
    model = models.GPT2(gpt2config)
    trainer_config = trainers.GPT2TrainerConfig()
    trainer = trainers.GPT2Trainer(trainer_config, model)
    """
    a simple example:
    sentence_id and sentence_mask have the same shape of torch.size([sentence_number, token_number])
    fine tuning is same with pre_training, except that the mask of position of prompts is 0
    """
    sentence_id = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0],
                                [2, 5, 6, 7, 8, 3, 11, 10]])
    sentence_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1]])
    dataset = data.TensorDataset(sentence_id, sentence_mask)
    data_iter = data.DataLoader(dataset, batch_size=2, shuffle=True)
    trainer.train(data_iter)
