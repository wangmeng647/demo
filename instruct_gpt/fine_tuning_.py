import json

import torch
import models
import trainers
from torch.utils import data
from tokenizer import Tokenizer
from tokenizer import load_data


if __name__ == '__main__':

    prompt_path = 'data/prompts.json'
    ans_path = 'data/answers.json'
    model_pth = 'gpt2_fine_tuned'
    prompts, ans = load_data(prompt_path, ans_path, combined=False)
    prompts = prompts[:100]
    ans = ans[:100]
    tokenizer = Tokenizer()
    tokenized_prompts = tokenizer.tokenize(prompts)
    tokenized_ans = tokenizer.tokenize(ans)
    prompts_ans = tokenized_prompts + tokenized_ans
    tokenizer.build_vocab(prompts_ans)
    tokenizer.save_to_trained(model_pth)
    sentence_id, mask = tokenizer.encode_prompts_ans(tokenized_prompts, tokenized_ans)
    print(len(tokenizer))
    gpt2config = models.GPT2Config(vocab_size=len(tokenizer))
    gpt2config.save_to_trained(model_pth)
    model = models.GPT2(gpt2config)
    trainer_config = trainers.GPT2TrainerConfig(device='cuda', visdom_open=True)
    trainer = trainers.GPT2Trainer(trainer_config, model)
    sentence_ids = torch.tensor(sentence_id)
    sentence_mask = torch.tensor(mask)
    dataset = data.TensorDataset(sentence_ids, sentence_mask)
    data_iter = data.DataLoader(dataset, batch_size=20, shuffle=True)
    trainer.train(data_iter, model_pth=model_pth)
