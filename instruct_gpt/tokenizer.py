import json
import thulac
import collections
import jieba
import os


def load_data(prompts_path, ans_path=None, prompts_mx_len=64, ans_mx_len=256, combined=False):

    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    if ans_path is not None:
        with open(ans_path, 'r', encoding='utf-8') as f:
            ans = json.load(f)

    text_combined = []
    text_prompts = []
    text_ans = []

    # process prompt and ans
    if ans_path is not None:
        for i in range(len(prompts)):
            if len(ans[i]) < ans_mx_len and len(prompts[i]) < prompts_mx_len:
                text_combined.append(prompts[i] + ans[i])
                text_prompts.append(prompts[i])
                text_ans.append(ans[i])
        if combined:
            return text_combined
        return text_prompts, text_ans
    # process prompt only
    else:
        for i in range(len(prompts)):
            if len(prompts[i]) < prompts_mx_len:
                text_prompts.append(prompts[i])
        return text_prompts


class Tokenizer:

    def __init__(self, vocab_file=None, unique=None):
        self.lac = thulac.thulac(seg_only=True)
        self.jie_ba = jieba
        self.token_to_id = None
        self.id_to_token = None
        self.unk = '[UNK]'
        self.unk_id = 0
        self.unique = ['[PAD]', '[BOS]', '[EOS]']
        if unique is not None:
            self.unique = unique
        if vocab_file is not None:
            with open(vocab_file, 'r', encoding='utf-8') as file:
                tokens = file.readlines()
                self.token_to_id = {token.rstrip('\n'): i for i, token in enumerate(tokens)}
                self.id_to_token = [token.rstrip('\n') for token in tokens]

    def __len__(self):
        return len(self.id_to_token)

    def tokenize(self, text):
        if isinstance(text, str):
            tokenized_text = [list(self.jie_ba.cut(text))]
            return tokenized_text
        elif isinstance(text[0], list):
            tokenized_text = []
            for sub_bath in text:
                tokenized_text.append([list(self.jie_ba.cut(sentence)) for sentence in sub_bath])
            return tokenized_text
        else:
            tokenized_text = [list(self.jie_ba.cut(sentence)) for sentence in text]
            return tokenized_text

    def build_vocab(self, train_data, min_fre):
        if isinstance(train_data[0][0], list):
            tokens = [token for batch in train_data for sentence in batch for token in sentence]
        else:
            tokens = [token for sentence in train_data for token in sentence]
        corpus_counter = sorted(collections.Counter(tokens).items(), key=lambda x: x[1], reverse=True)
        vocab = [self.unk] + self.unique

        for token, num in corpus_counter:
            if num < min_fre:
                continue
            vocab.append(token)

        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = vocab

    def encode_text(self, text):
        """
        there is no truncate, using the longest sentence as max_len if the text has more than 1 sentence
        :param text: can be list or list of list
        :return: encoded_ids and mask
        """
        encoded_ids = []
        encoded_mask = []

        max_len = max([len(sentence) for sentence in text])
        for sentence in text:
            processed_sentence = sentence + ['[EOS]'] + ['[PAD]'] * (max_len - len(sentence))
            encoded_ids.append([self.token_to_id.get(token, self.unk_id) for token in processed_sentence])
            encoded_mask.append([1] * (len(sentence) + 1) + [0] * (max_len - len(sentence)))

        return encoded_ids, encoded_mask

    def encode_prompts_ans(self, prompts, ans):
        """
        there is no truncate, using the longest length of combination of prompt and ans as the max_len if has more than
         1 prompt.
         the mask of prompt is 0
        :param prompts: can be list or list of list
        :param ans: can be list or list of list
        :return: encoded_ids and mask
        """
        encoded_ids = []
        encoded_mask = []

        if isinstance(prompts[0], list):
            max_len = max([len(prompts[i]) + len(ans[i]) for i in range(len(prompts))])
            for i in range(len(prompts)):
                len_prompts = len(prompts[i])
                len_ans = len(ans[i])
                processed_sentence = prompts[i] + ans[i] + ['[EOS]'] + ['[PAD]'] * (max_len - len_prompts - len_ans)
                encoded_ids.append([self.token_to_id.get(token, self.unk_id) for token in processed_sentence])
                encoded_mask.append([0] * len_prompts + [1] * (len_ans + 1) + [0] * (max_len - len_prompts - len_ans))
        else:
            len_prompts = len(prompts)
            len_ans = len(ans)
            processed_sentence = prompts + ans + ['[EOS]']
            encoded_ids.append([self.token_to_id.get(token, self.unk_id) for token in processed_sentence])
            encoded_mask.append([0] * len_prompts + [1] * (len_ans + 1))

        return encoded_ids, encoded_mask

    def encode_rank(self, text):
        """
        encode the ranked sentence, no truncate, mx_len is the longest sentence
        :param text: list of list of list, the second dimension of text is the rank of sentence
        :return: encoded text
        """
        max_len = max([len(sentence) for batch in text for sentence in batch])
        encoded_ids, mask = [], []
        for batch in text:
            batch_encoded = []
            batch_mask = []
            for sentence in batch:
                add_eos = sentence + ['[EOS]'] + ['[PAD]'] * (max_len - len(sentence))
                batch_encoded.append([self.token_to_id.get(token, self.unk_id) for token in add_eos])
                batch_mask.append([1] * (len(sentence) + 1) + [0] * (max_len - len(sentence)))
            encoded_ids.append(batch_encoded)
            mask.append(batch_mask)
        return encoded_ids, mask

    def ranked_id_padding(self, sentence_ids):
        """
        padding the ranked sentence_id, no truncate, mx_len is the longest sentence
        :param sentence_ids: list of list of list, the second dimension of text is the rank of sentence
        :return: sentence_id
        """
        max_len = max([len(sentence) for batch in sentence_ids for sentence in batch])
        batch, rank = len(sentence_ids), len(sentence_ids[0])
        mask = [[] for _ in range(batch)]
        for batch_i in range(batch):
            for rank_j in range(rank):
                len_i_j = len(sentence_ids[batch_i][rank_j])
                sentence_ids[batch_i][rank_j].append(self.token_to_id['[EOS]'])
                if len_i_j == max_len:
                    mask[batch_i].append([1] * (max_len + 1))
                else:
                    mask[batch_i].append([1] * (len_i_j + 1) + [0] * (max_len - len_i_j))
                    sentence_ids[batch_i][rank_j] += [self.token_to_id['[PAD]']] * (max_len - len_i_j)
        return sentence_ids, mask

    def decode(self, text):

        decoded_text = []

        if isinstance(text[0], list):
            for sentence_ids in text:
                decoded_text.append([self.id_to_token[token_id] for token_id in sentence_ids])
        else:
            decoded_text.append([self.id_to_token[token_id] for token_id in text])

        for i in range(len(decoded_text)):
            decoded_text[i] = ''.join(decoded_text[i])

        return decoded_text

    def save_to_file(self, save_path, encoding='utf-8'):
        with open(save_path, 'w', encoding=encoding) as file:
            for token in self.token_to_id:
                file.write(token + '\n')

    def save_to_trained(self, model_path, encoding='utf-8'):

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        with open(model_path + '/' + 'vocab.txt', 'w', encoding=encoding) as file:
            for token in self.token_to_id:
                file.write(token + '\n')

    @classmethod
    def load_from_trained(cls, pth):
        return cls(pth + '/' + 'vocab.txt')

'''prompt_path = 'data\\prompts.json'
ans_path = 'data\\answers.json'
vocab_file = 'data\\vocab.txt'
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompts = json.load(f)
with open(ans_path, 'r', encoding='utf-8') as f:
    ans = json.load(f)

pro_filtered = []
ans_filtered = []
for i in range(len(prompts)):
    if len(prompts[i]) > 1000 or len(ans[i]) > 2000:
        continue
    pro_filtered.append(prompts[i])
    ans_filtered.append(ans[i])
tokenizer = Tokenizer()
tokenized_pro = tokenizer.tokenize(pro_filtered[:2])
tokenized_ans = tokenizer.tokenize(ans_filtered[:2])
text = tokenized_pro + tokenized_ans
tokenizer.build_vocab(text)
s_id, mask = tokenizer.encode_prompts_ans(tokenized_pro, tokenized_ans)
print(len(tokenizer))
tokenizer.save_vocab(vocab_file)'''
