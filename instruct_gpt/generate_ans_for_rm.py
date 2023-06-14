import torch
import models
import trainers
from torch.utils import data
from tokenizer import Tokenizer
from tokenizer import load_data


if __name__ == '__main__':
    prompts = ['打完', '不要怕']
    file = 'gpt2_fine_tuned'
    tokenizer = Tokenizer.load_from_trained(file)
    gpt2_config = models.GPT2Config.load_from_trained(file)
    gpt2_config.vocab_size = len(tokenizer)
    model = models.GPT2(gpt2_config)
    model.load_from_trained(file)
    gpt2_generator_config = models.GPT2SeqGeneratorConfig(model,
                                                          tokenizer,
                                                          top_k=3,
                                                          generate_pattern='top_k_hierarchy'
                                                          )
    gpt2_generator = models.GPT2SeqGenerator(gpt2_generator_config)
    hierarchy = 3
    rank_ans = [[] for _ in range(len(prompts))]
    for i in range(len(prompts)):
        for j in range(hierarchy):
            gpt2_generator.config.hierarchy = j + 1
            ans = gpt2_generator.generate(prompts[i], only_id=True)
            rank_ans[i].append(ans)
    print(rank_ans)
