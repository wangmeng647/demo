import torch
import models
import trainers
from torch.utils import data


if __name__ == '__main__':

    gpt2config = models.GPT2Config(vocab_size=100)
    rl_model = models.GPT2(gpt2config)
    vf_model = models.GPT2RM(gpt2config)
    ppo_trainer_config = trainers.PPOTrainerConfig(entropy_loss=True)
    ppo_trainer = trainers.PPOTrainer(ppo_trainer_config, rl_model, vf_model)
    """
    a simple example:
    sentence_id and sentence_mask have the same shape of torch.size([sentence_number, token_number])
    sentence_reward is the reward for each sentence shape torch.size([sentence_number])
    note: this model exclude entropy loss described in ppo algorithm paper, can include it in PPOTrainerConfig
    """
    sentence_id = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0],
                                [2, 5, 6, 7, 8, 3, 11, 10]])
    sentence_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 1, 1]])
    sentence_reward = torch.tensor([1, -1])
    dataset = data.TensorDataset(sentence_id, sentence_mask, sentence_reward)
    data_iter = data.DataLoader(dataset, batch_size=2, shuffle=True)
    ppo_trainer.train(data_iter)
