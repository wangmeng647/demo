import torch
import models
import trainers
from torch.utils import data
from tokenizer import Tokenizer
from tokenizer import load_data
from torch import nn
import fuc_tools

if __name__ == '__main__':

    text_path = 'data/text.json'
    model_pth = 'gpt2_pre_trained'
    vocab_pth = 'data/vocab.txt'
    text = load_data(text_path, prompts_mx_len=196)
    text = text[:10000]
    tokenizer = Tokenizer(vocab_pth)
    tokenized_text = tokenizer.tokenize(text)
    sentence_id, mask = tokenizer.encode_text(tokenized_text)
    print(len(sentence_id[0]))
    gpt2config = models.GPT2Config(vocab_size=len(tokenizer), sentence_max_len=len(sentence_id[0]))
    # save model config
    #gpt2config.save_to_trained(model_pth)
    model = models.GPT2(gpt2config)
    trainer_config = trainers.GPT2TrainerConfig(epochs=1000,
                                                device='cuda',
                                                visdom_open=True,
                                                epoch_num_to_save=30,
                                                warmup=fuc_tools.warmup(warmup_steps=10, total_steps=1000)
                                                )
    optimizer = torch.optim.Adam(model.parameters())
    trainer = trainers.GPT2Trainer(trainer_config, model, optimizer=optimizer)
    sentence_ids = torch.tensor(sentence_id)
    sentence_mask = torch.tensor(mask)
    dataset = data.TensorDataset(sentence_ids, sentence_mask)
    data_iter = data.DataLoader(dataset, batch_size=90, shuffle=True)
    trainer.train(data_iter, model_pth=model_pth)

