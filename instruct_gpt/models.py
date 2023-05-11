import torch
from torch import nn
import tools
import math
import collections
import torch.nn.functional as F


def masked_softmax(softmax_input, valid_len=None):
    """softmax with mask if valid_len is not None, else raw softmax"""

    if valid_len is None:
        return torch.softmax(softmax_input, dim=-1)
    else:
        shape = softmax_input.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, shape[1])
        else:
            valid_len = valid_len.reshape(-1)

        softmax_input = softmax_input.reshape(-1, shape[-1])
        mask = torch.arange(shape[-1], device=softmax_input.device)[None, :] < valid_len[:, None]
        softmax_input[~mask] = -1e6
        return torch.softmax(softmax_input.reshape(shape), dim=-1)


def transpose_attention_input(attention_input, num_heads):
    """process transpose for multi_head attention"""
    shape = attention_input.shape
    attention_input = attention_input.reshape(shape[0], shape[1], num_heads, -1)
    attention_input = attention_input.permute(0, 2, 1, 3)
    return attention_input.reshape(-1, attention_input.shape[2], attention_input.shape[3])


def transpose_attention_output(attention_output, num_heads):
    """process transpose for heads concatenate"""
    shape = attention_output.shape
    attention_output = attention_output.reshape(-1, num_heads, shape[1], shape[-1])
    attention_output = attention_output.permute(0, 2, 1, 3)
    return attention_output.reshape(attention_output.shape[0], shape[1], -1)


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(config.hidden_dim, config.ff_affine_dim)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(config.ff_affine_dim, config.hidden_dim)

    def forward(self, input_id):
        return self.linear_layer_2(self.relu(self.linear_layer_1(input_id)))


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super(GPT2MLP, self).__init__()
        self.linear_layer_1 = nn.Linear(config.hidden_dim, config.ff_affine_dim)
        self.gelu = nn.GELU()
        self.linear_layer_2 = nn.Linear(config.ff_affine_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_id):
        return self.dropout(self.linear_layer_2(self.gelu(self.linear_layer_1(input_id))))


class AddNorm(nn.Module):
    def __init__(self, config):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, layer_input, layer_output):
        return self.norm(layer_input + self.dropout(layer_output))


class DotProductAttention(nn.Module):

    @staticmethod
    def forward(q, k, v, valid_len):
        d_k = torch.tensor(q.shape[-1])
        score = torch.bmm(q, k.transpose(2, 1)) / torch.sqrt(d_k)
        attention_weights = masked_softmax(score, valid_len)
        return torch.bmm(attention_weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_head = config.num_attention_head
        self.dot_product_attention = DotProductAttention()
        self.q_affine = nn.Linear(config.hidden_dim, config.multi_head_affine_dim)
        self.k_affine = nn.Linear(config.hidden_dim, config.multi_head_affine_dim)
        self.v_affine = nn.Linear(config.hidden_dim, config.multi_head_affine_dim)
        self.out_affine = nn.Linear(config.multi_head_affine_dim, config.hidden_dim)

    def forward(self, q, k, v, valid_len):
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len, self.num_attention_head, dim=0)
        q = transpose_attention_input(self.q_affine(q), self.num_attention_head)
        k = transpose_attention_input(self.k_affine(k), self.num_attention_head)
        v = transpose_attention_input(self.v_affine(v), self.num_attention_head)
        output = self.dot_product_attention(q, k, v, valid_len)
        concat_attention = transpose_attention_output(output, self.num_attention_head)
        return self.out_affine(concat_attention)


class EncoderPositionalEncoding(nn.Module):
    def __init__(self, config):
        super(EncoderPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(config.vocab_size_encoder, config.hidden_dim)

        if config.abs_pos_encode:
            self.pos_code = torch.zeros((1, config.position_encoding_max_len, config.hidden_dim))
            abs_position_parameter = torch.arange(
                                                 config.position_encoding_max_len
                                                 ).reshape(-1, 1) / torch.pow(
                                                                              10000, torch.arange(0, config.hidden_dim,
                                                                                                  2
                                                                                                  ) / config.hidden_dim
                                                                             )
            self.pos_code[:, :, 0::2] = torch.sin(abs_position_parameter)
            self.pos_code[:, :, 1::2] = torch.cos(abs_position_parameter)
        else:
            self.pos_code = nn.Parameter(torch.rand(1, config.position_encoding_max_len, config.hidden_dim))

        self.sqrt_dim = 1
        if config.embedding_sqrt_mul:
            self.sqrt_dim = torch.sqrt(torch.tensor(config.hidden_dim))

    def forward(self, input_id):
        encoded_ip = self.embedding(input_id) * self.sqrt_dim + \
                     self.pos_code[:, input_id.shape[1], :].to(input_id.device)
        return self.dropout(encoded_ip)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)
        self.add_norm_attention = AddNorm(config)
        self.add_norm_feedforward = AddNorm(config)

    def forward(self, input_id, input_valid_len):
        output_attention = self.multi_head_attention(input_id, input_id, input_id, input_valid_len)
        output_add_norm = self.add_norm_attention(input_id, output_attention)
        output_feedforward = self.feedforward(output_add_norm)
        return self.add_norm_feedforward(output_add_norm, output_feedforward)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.positional_embedding = EncoderPositionalEncoding(config)
        self.block_stack = nn.Sequential()

        for i in range(config.block_num):
            self.block_stack.add_module('block' + str(i), EncoderBlock(config))

    def forward(self, input_id, valid_len=None):
        input_id = self.positional_embedding(input_id)
        for block in self.block_stack:
            input_id = block(input_id, valid_len)
        return input_id


class DecoderPositionalEncoding(nn.Module):
    def __init__(self, config, shared_embedding_matrix=None):
        super(DecoderPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout)

        if config.embedding_share:
            self.embedding = nn.Embedding(config.vocab_size_decoder, config.hidden_dim, _weight=shared_embedding_matrix)
        else:
            self.embedding = nn.Embedding(config.vocab_size_decoder, config.hidden_dim)

        if config.abs_pos_encode:
            self.pos_code = torch.zeros((1, config.position_encoding_max_len, config.hidden_dim))
            abs_position_parameter = torch.arange(
                                                 config.position_encoding_max_len
                                                 ).reshape(-1, 1) / torch.pow(
                                                                              10000, torch.arange(0, config.hidden_dim,
                                                                                                  2
                                                                                                  ) / config.hidden_dim
                                                                             )
            self.pos_code[:, :, 0::2] = torch.sin(abs_position_parameter)
            self.pos_code[:, :, 1::2] = torch.cos(abs_position_parameter)
        else:
            self.pos_code = nn.Parameter(torch.rand(1, config.position_encoding_max_len, config.hidden_dim))

        self.sqrt_dim = 1
        if config.embedding_sqrt_mul:
            self.sqrt_dim = torch.sqrt(torch.tensor(config.hidden_dim))

    def forward(self, input_ids):
        encoded_ids = self.embedding(input_ids) * self.sqrt_dim + \
                      self.pos_code[:, :input_ids.shape[1], :].to(input_ids.device)
        return self.dropout(encoded_ids)


class GPT2PositionalEncoding(nn.Module):
    def __init__(self, config, shared_embedding_matrix=None):
        super(GPT2PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout)

        if config.embedding_share:
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim, _weight=shared_embedding_matrix)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        if config.abs_pos_encode:
            self.pos_code = torch.zeros((1, config.position_encoding_max_len, config.hidden_dim))
            abs_position_parameter = torch.arange(
                                                 config.position_encoding_max_len
                                                 ).reshape(-1, 1) / torch.pow(
                                                                              10000, torch.arange(0, config.hidden_dim,
                                                                                                  2
                                                                                                  ) / config.hidden_dim
                                                                             )
            self.pos_code[:, :, 0::2] = torch.sin(abs_position_parameter)
            self.pos_code[:, :, 1::2] = torch.cos(abs_position_parameter)
        else:
            self.pos_code = nn.Parameter(torch.rand(1, config.position_encoding_max_len, config.hidden_dim))

        self.sqrt_dim = 1
        if config.embedding_sqrt_mul:
            self.sqrt_dim = torch.sqrt(torch.tensor(config.hidden_dim))

    def forward(self, input_ids):
        len_idx = input_ids.shape[1]
        encoded_ids = self.embedding(input_ids) * self.sqrt_dim + self.pos_code[:, len_idx, :].to(input_ids.device)
        return self.dropout(encoded_ids)


class DecoderBlock(nn.Module):
    def __init__(self, config, i):
        super(DecoderBlock, self).__init__()
        self.attention_self = MultiHeadAttention(config)
        self.add_norm_attention_self = AddNorm(config)
        self.attention_cross = MultiHeadAttention(config)
        self.add_norm_attention_cross = AddNorm(config)
        self.feedforward = FeedForward(config)
        self.add_norm_feedforward = AddNorm(config)
        self.i = i

    def forward(self, input_id, state):
        # state: (encoder_input, encoder_valid_len, [key_value_block_i, key_value_block_i + 1, ...)
        encoder_output, encoder_valid_len = state[0], state[1]

        if self.training:
            batch_size, max_len, _ = input_id.shape
            attention_mask_len = torch.arange(1, 1 + max_len, device=input_id.device).repeat(batch_size, 1)
        else:
            attention_mask_len = None

        if state[2][self.i] is None:
            attention_k_v = input_id
        else:
            attention_k_v = torch.cat((state[2][self.i], input_id), dim=1)

        state[2][self.i] = attention_k_v
        output_attention_self = self.attention_self(input_id, attention_k_v, attention_k_v, attention_mask_len)
        output_add_norm_attention_self = self.add_norm_attention_self(input_id, output_attention_self)
        output_attention_cross = self.attention_cross(output_add_norm_attention_self,
                                                      encoder_output,
                                                      encoder_output,
                                                      encoder_valid_len
                                                      )
        output_add_norm_attention_cross = self.add_norm_attention_cross(output_add_norm_attention_self,
                                                                        output_attention_cross
                                                                        )
        output_decoder_block = self.add_norm_feedforward(output_add_norm_attention_cross,
                                                         self.feedforward(output_add_norm_attention_cross)
                                                         )
        return output_decoder_block, state


class GPT2Block(nn.Module):
    def __init__(self, config, i):
        super(GPT2Block, self).__init__()

        self.config = config
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attention = MultiHeadAttention(config)
        self.drop_out = nn.Dropout(config.dropout)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = GPT2MLP(config)
        self.i = i

    def forward(self, input_ids, state):
        # state: ([key_value_block_i, key_value_block_i + 1, ...)

        ln_1_output = self.ln_1(input_ids)

        batch_size, max_len, _ = ln_1_output.shape
        attention_mask_len = torch.arange(1, 1 + max_len, device=ln_1_output.device).repeat(batch_size, 1)

        if state is None:
            attention_k_v = ln_1_output
        else:
            if state[self.i] is None:
                attention_k_v = ln_1_output
            else:
                attention_k_v = torch.cat((state[self.i], ln_1_output), dim=1)
            state[self.i] = attention_k_v

        output_attention = self.attention(ln_1_output, attention_k_v, attention_k_v, attention_mask_len)
        output_drop_out = self.drop_out(output_attention)
        add_1 = input_ids + output_drop_out
        ln_2_output = self.ln_2(add_1)
        output_mlp = self.mlp(ln_2_output)
        add_2 = add_1 + output_mlp
        return add_2, state


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()

        self.embedding_share = config.embedding_share
        if config.embedding_share:
            self.shared_embedding_matrix = nn.Parameter(torch.rand(config.vocab_size_decoder, config.hidden_dim))
            self.positional_embedding = DecoderPositionalEncoding(config,
                                                                  shared_embedding_matrix=self.shared_embedding_matrix
                                                                  )
        else:
            self.un_embedding = nn.Linear(config.hidden_dim, config.vocab_size_decoder, bias=False)
            self.positional_embedding = DecoderPositionalEncoding(config)
        self.block_stack = nn.Sequential()
        self.block_num = config.block_num

        for i in range(config.block_num):
            self.block_stack.add_module('block' + str(i), DecoderBlock(config, i))

    def init_state(self, enc_output, enc_valid_len):
        return [enc_output, enc_valid_len, [None] * self.block_num]

    def forward(self, input_id, state):
        input_id = self.positional_embedding(input_id)

        for blk in self.block_stack:
            input_id, state = blk(input_id, state)

        if self.embedding_share:
            return torch.matmul(input_id, self.shared_embedding_matrix.transpose(1, 0)), state
        else:
            return self.un_embedding(input_id), state


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.loss = MaskedSoftmaxCELoss()

    def forward(self, encoder_input, encoder_valid_len, decoder_input, tgt_vocab, decoder_valid_len=None):
        encoder_output = self.encoder(encoder_input, encoder_valid_len)
        decoder_state = self.decoder.init_state(encoder_output, encoder_valid_len)

        bos = torch.tensor([tgt_vocab['<bos>']] * decoder_input.shape[0], device=encoder_input.device).reshape(-1, 1)
        input_id = torch.cat([bos, decoder_input[:, :-1]], 1)
        logits, decoder_state = self.decoder(input_id, decoder_state)
        loss = self.loss(logits, decoder_input, decoder_valid_len)

        return loss


class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.positional_embedding = GPT2PositionalEncoding(config)
        self.block_stack = nn.Sequential()
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.un_embedding = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.loss = GPT2MaskedSoftmaxCELoss()

        self.block_num = config.block_num
        for i in range(config.block_num):
            self.block_stack.add_module('block' + str(i), GPT2Block(config, i))

    def init_state(self, prompt=None):
        if prompt is None:
            return [None] * self.block_num
        else:
            state = [None] * self.block_num
            _, _, state = self.forward(prompt, state)
            return state

    def forward(self, input_ids, input_mask=None, state=None):
        label = input_ids[:, 1:]
        input_ids = self.positional_embedding(input_ids)

        for blk in self.block_stack:
            input_ids, state = blk(input_ids, state)
        output_hidden = self.ln(input_ids)
        logits = self.un_embedding(output_hidden)
        shifted_logits = logits[:, :-1, :]
        if input_mask is not None:
            input_mask = input_mask[:, 1:]
        loss = self.loss(shifted_logits, label, input_mask)
        model_output = {'logits': logits,
                        'output_hidden': output_hidden,
                        'loss': loss,
                        'state': state
                        }
        return model_output


class GPT2RM(nn.Module):
    def __init__(self, config):
        super(GPT2RM, self).__init__()
        self.gpt2 = GPT2(config)
        self.value_head_hidden_dim = nn.Linear(config.hidden_dim, 1)
        self.value_head_seq_len_dim = nn.Linear(config.sentence_max_len, 1)

    def forward(self, input_ids, input_mask=None):
        output_hidden = self.gpt2(input_ids, input_mask)['output_hidden']
        affine_1 = self.value_head_hidden_dim(output_hidden)
        if input_mask is None:
            input_mask = 1
        masked_affine = affine_1.squeeze(dim=-1) * input_mask
        return self.value_head_seq_len_dim(masked_affine)

    def vf_forward(self, input_ids, input_mask=None):
        output_hidden = self.gpt2(input_ids, input_mask)['output_hidden']
        affine_1 = self.value_head_hidden_dim(output_hidden)

        return affine_1.squeeze(dim=-1)

    @staticmethod
    def rank_reward_loss(batch_reward_table):
        batch_size, sentence_len = batch_reward_table.shape
        loss = 0
        combination_nums = math.comb(sentence_len, 2)
        for sentence_pairs in batch_reward_table:
            for sentence_i in range(sentence_len - 1):
                for sentence_j in range(sentence_i + 1, sentence_len):
                    loss -= F.logsigmoid(sentence_pairs[sentence_i] - sentence_pairs[sentence_j])

        return loss / combination_nums


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self):
        super(MaskedSoftmaxCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, label_predicted, label_true, valid_len=None):
        if valid_len is None:
            loss = self.ce_loss(label_predicted.permute(0, 2, 1), label_true)
        else:
            weights = torch.ones_like(label_true)
            max_len = weights.shape[1]
            mask = torch.arange(max_len, device=label_true.device)[None, :] < valid_len[:, None]
            weights[~mask] = 0
            unweighted_loss = self.ce_loss(label_predicted.permute(0, 2, 1), label_true)
            loss = unweighted_loss * weights

        return loss.sum()


class GPT2MaskedSoftmaxCELoss(nn.Module):
    def __init__(self):
        super(GPT2MaskedSoftmaxCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, label_predicted, label_true, input_mask=None):
        if input_mask is None:
            loss = self.ce_loss(label_predicted.permute(0, 2, 1), label_true)
        else:
            unmasked_loss = self.ce_loss(label_predicted.permute(0, 2, 1), label_true)
            loss = unmasked_loss * input_mask

        return loss.sum()


'''def target_predict(source):
    predict = []
    for eng in source:
        eng = ' '.join(eng)
        translation, _ = predict_seq2seq(net, eng, src_vocab, tgt_vocab, max_len, device)
        predict.append(translation.split())
    return predict'''


def bleu(predicted_seq, label_seq, k_grams):
    """calculate bleu score for predicted sentence"""
    predicted_tokens, label_tokens = predicted_seq.split(' '), label_seq.split(' ')
    len_predicted, len_label = len(predicted_tokens), len(label_tokens)
    score = math.exp(min(0., 1 - len_label / len_predicted))

    for n in range(1, k_grams + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)

        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1

        for i in range(len_predicted - n + 1):
            if label_subs[' '.join(predicted_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(predicted_tokens[i: i + n])] -= 1

        score *= math.pow(num_matches / (len_predicted - n + 1), math.pow(0.5, n))

    return score


class TransformerGeneratorConfig:
    def __init__(self):
        pass


class Tokenizer:
    def __init__(self):
        pass


class TransformerSeqGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
        model = self.config.model
        model.eval()
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        encoder_valid_len = torch.tensor([len(src_tokens)], device=self.config.device)
        src_tokens = tools.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
        encoder_input_id = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=self.config.device), dim=0)
        enc_outputs = model.encoder(encoder_input_id, encoder_valid_len)
        decoder_state = model.decoder.init_state(enc_outputs, encoder_valid_len)
        decoder_input = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                        dtype=torch.long,
                                        device=self.config.device),
                                        dim=0
                                        )
        output_seq, attention_weight_seq = [], []
        for _ in range(num_steps):

            decoder_output, decoder_state = model.decoder(decoder_input, decoder_state)
            decoder_input = decoder_output.argmax(dim=2)
            token_pre = decoder_input.squeeze(dim=0).type(torch.int32).item()
            if save_attention_weights:
                attention_weight_seq.append(model.decoder.attention_weights)
            if token_pre == tgt_vocab['<eos>']:
                break
            output_seq.append(token_pre)
        return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


class GPT2SeqGeneratorConfig:
    def __init__(
        self,
        model,
        vocab,
        output_max_len=10,
        save_attention_weights=False,
        device='cpu'
    ):
        self.model = model
        self.vocab = vocab
        self.output_max_len = output_max_len
        self.save_attention_weights = save_attention_weights
        self.device = device


class GPT2SeqGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, input_sentence):
        vocab = self.config.vocab
        model = self.config.model
        model.eval()

        token_ids = vocab[input_sentence.lower().split(' ')]
        input_ids = torch.unsqueeze(torch.tensor(token_ids, dtype=torch.long, device=self.config.device), dim=0)

        output_seq, attention_weight_seq = [], []
        for i in range(self.config.output_max_len):

            logits = model(input_ids)['logits']
            token_predicted = logits[0, -1, :].argmax()
            input_ids = torch.cat((input_ids, token_predicted.unsqueeze(dim=-1).unsqueeze(dim=-1)), dim=-1)
            #token_pre = decoder_input.squeeze(dim=0).type(torch.int32).item()
            if self.config.save_attention_weights:
                attention_weight_seq.append(model.decoder.attention_weights)
            if token_predicted == vocab['<eos>']:
                break
            output_seq.append(token_predicted)
        return ' '.join(vocab.to_tokens(output_seq)), attention_weight_seq


class GPT2Config:
    def __init__(
        self,
        hidden_dim=32,
        block_num=2,
        num_attention_head=4,
        q_size=32,
        k_size=32,
        v_size=32,
        multi_head_affine_dim=32,
        ff_affine_dim=64,
        dropout=0.1,
        batch_size=64,
        sentence_max_len=10,
        position_encoding_max_len=1000,
        vocab_size=None,
        embedding_share=False,
        embedding_sqrt_mul=False,
        abs_pos_encode=False
    ):
        if vocab_size is None:
            raise 'please assign vocab_size'
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.num_attention_head = num_attention_head
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.multi_head_affine_dim = multi_head_affine_dim
        self.ff_affine_dim = ff_affine_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.sentence_max_len = sentence_max_len
        self.position_encoding_max_len = position_encoding_max_len
        self.embedding_share = embedding_share
        self.embedding_sqrt_mul = embedding_sqrt_mul
        self.abs_pos_encode = abs_pos_encode
        self.vocab_size = vocab_size


class TransformerConfig:
    def __init__(
        self,
        hidden_dim=32,
        block_num=2,
        num_attention_head=4,
        q_size=32,
        k_size=32,
        v_size=32,
        multi_head_affine_dim=32,
        ff_affine_dim=64,
        dropout=0.1,
        batch_size=64,
        sentence_max_len=10,
        position_encoding_max_len=1000,
        vocab_size_encoder=None,
        vocab_size_decoder=None,
        embedding_share=True,
        embedding_sqrt_mul=True,
        abs_pos_encode=True
    ):
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.num_attention_head = num_attention_head
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.multi_head_affine_dim = multi_head_affine_dim
        self.ff_affine_dim = ff_affine_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.sentence_max_len = sentence_max_len
        self.position_encoding_max_len = position_encoding_max_len
        self.embedding_share = embedding_share
        self.embedding_sqrt_mul = embedding_sqrt_mul
        self.abs_pos_encode = abs_pos_encode
        self.vocab_size_encoder = vocab_size_encoder
        self.vocab_size_decoder = vocab_size_decoder


'''def average_bleu(source, target, k):
    target_pre = target_predict(source)
    total = 1e-6
    c = 0
    for pre, tgt in zip(target_pre, target):
        score = bleu(' '.join(pre), ' '.join(tgt), k)
        c += 1
        total += score
    return total / c'''

