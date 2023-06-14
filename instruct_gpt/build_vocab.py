from tokenizer import Tokenizer
from tokenizer import load_data


prompt_path = 'data/prompts.json'
ans_path = 'data/answers.json'
text_path = 'data/text.json'
model_pth = 'data'
prompts, ans = load_data(prompt_path, ans_path, prompts_mx_len=64, ans_mx_len=128, combined=False)
text = load_data(text_path, prompts_mx_len=196)
prompts = prompts[:5000]
ans = ans[:5000]
text = text[:10000]
tokenizer = Tokenizer()
tokenized_prompts = tokenizer.tokenize(prompts)
tokenized_ans = tokenizer.tokenize(ans)
tokenized_text = tokenizer.tokenize(text)
total_sentences = tokenized_prompts + tokenized_ans + tokenized_text
tokenizer.build_vocab(total_sentences, min_fre=3)
tokenizer.save_to_trained(model_pth)
print(len(tokenizer))