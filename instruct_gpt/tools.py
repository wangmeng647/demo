# coding: utf-8
import numpy as np
from torch.utils import data
import collections
import torch


def synthetic_data(seed=1984):
    np.random.seed(seed)
    n = 100  # 各类的样本数
    dim = 2  # 数据的元素个数
    cls_num = 3  # 类别数

    x = np.zeros((n * cls_num, dim))
    t = np.zeros((n * cls_num, cls_num), dtype=np.int)

    for j in range(cls_num):
        for i in range(n):  # N*j, N*(j+1)):
            rate = i / n
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = n * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))
    

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)        


class Vocab:
    def __init__(self, tokens=None, min_freq=2, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True
                                  )
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_data_nmt(batch_size, sentence_max_len, file_path, num_examples=600):
    text = preprocess_nmt(read_data_nmt(file_path))
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, sentence_max_len)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, sentence_max_len)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab, source, target


def load_data_gpt(batch_size, sentence_max_len, file_path, num_examples=600):
    text = preprocess_nmt(read_data_nmt(file_path))
    source, _ = tokenize_nmt(text, num_examples)
    vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, vocab, sentence_max_len)
    data_arrays = (src_array, src_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, vocab, source


def preprocess_nmt(text):

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # add space between char and punctuation
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]

    return ''.join(out)


def read_data_nmt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def assign_type(x, *args, **kwargs):
    return x.type(*args, **kwargs)


def reduce_sum(x, *args, **kwargs):
    return x.sum(*args, **kwargs)


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[i] for i in lines]
    lines = [i + [vocab['<eos>']] for i in lines]
    array = torch.tensor([truncate_pad(i, num_steps, vocab['<pad>']) for i in lines])
    valid_len = reduce_sum(assign_type(array != vocab['<pad>'], torch.int32), 1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_x = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_x = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state)
        dec_x = y.argmax(dim=2)
        predict = dec_x.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if predict == tgt_vocab['<eos>']:
            break
        output_seq.append(predict)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


