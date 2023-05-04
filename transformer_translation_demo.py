import torch
from torch import nn
import tools
import math
from visdom import Visdom
import time
from matplotlib import pyplot as plt
import collections


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


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


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(config.hidden_dim, config.ff_affine_dim)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(config.ff_affine_dim, config.hidden_dim)

    def forward(self, input_id):
        return self.linear_layer_2(self.relu(self.linear_layer_1(input_id)))


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
            self.pos_code = torch.zeros((1, config.sentence_max_len, config.hidden_dim))
            abs_position_parameter = torch.arange(
                                                 config.sentence_max_len
                                                 ).reshape(-1, 1) / torch.pow(
                                                                              10000, torch.arange(0, config.hidden_dim,
                                                                                                  2
                                                                                                  ) / config.hidden_dim
                                                                             )
            self.pos_code[:, :, 0::2] = torch.sin(abs_position_parameter)
            self.pos_code[:, :, 1::2] = torch.cos(abs_position_parameter)
        else:
            self.pos_code = nn.Parameter(torch.rand(1, config.sentence_max_len, config.hidden_dim))

        self.sqrt_dim = 1
        if config.embedding_sqrt_mul:
            self.sqrt_dim = torch.sqrt(torch.tensor(config.hidden_dim))

    def forward(self, input_id):
        encoded_ip = self.embedding(input_id) * self.sqrt_dim + self.pos_code.to(input_id.device)
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
            self.pos_code = torch.zeros((1, config.sentence_max_len, config.hidden_dim))
            abs_position_parameter = torch.arange(
                                                 config.sentence_max_len
                                                 ).reshape(-1, 1) / torch.pow(
                                                                              10000, torch.arange(0, config.hidden_dim,
                                                                                                  2
                                                                                                  ) / config.hidden_dim
                                                                             )
            self.pos_code[:, :, 0::2] = torch.sin(abs_position_parameter)
            self.pos_code[:, :, 1::2] = torch.cos(abs_position_parameter)
        else:
            self.pos_code = nn.Parameter(torch.rand(1, config.sentence_max_len, config.hidden_dim))

        self.sqrt_dim = 1
        if config.embedding_sqrt_mul:
            self.sqrt_dim = torch.sqrt(torch.tensor(config.hidden_dim))

    def forward(self, input_id):
        encoded_ip = self.embedding(input_id) * self.sqrt_dim + self.pos_code.to(input_id.device)
        return self.dropout(encoded_ip)


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
            decoder_valid_len = torch.arange(1, 1 + max_len, device=input_id.device).repeat(batch_size, 1)
        else:
            decoder_valid_len = None

        if state[2][self.i] is None:
            attention_k_v = input_id
        else:
            attention_k_v = torch.cat((state[2][self.i], input_id), dim=1)

        state[2][self.i] = attention_k_v
        output_attention_self = self.attention_self(input_id, attention_k_v, attention_k_v, decoder_valid_len)
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
        self.loss = MaskedSoftmaxCELoss(config.loss_mean)

    def forward(self, encoder_input, encoder_valid_len, decoder_input, decoder_valid_len=None):
        encoder_output = self.encoder(encoder_input, encoder_valid_len)
        decoder_state = self.decoder.init_state(encoder_output, encoder_valid_len)

        bos = torch.tensor([tgt_vocab['<bos>']] * decoder_input.shape[0], device=encoder_input.device).reshape(-1, 1)
        input_id = torch.cat([bos, decoder_input[:, :-1]], 1)
        reversed_embedding_output, decoder_state = self.decoder(input_id, decoder_state)
        loss = self.loss(reversed_embedding_output, decoder_input, decoder_valid_len)
        return loss


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self, mean=False):
        super(MaskedSoftmaxCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mean = mean
    def forward(self, label_predicted, label_true, valid_len):
        weights = torch.ones_like(label_true)
        max_len = weights.shape[1]
        mask = torch.arange(max_len, device=label_true.device)[None, :] < valid_len[:, None]
        weights[~mask] = 0
        unweighted_loss = self.ce_loss(label_predicted.permute(0, 2, 1), label_true)
        weighted_loss = unweighted_loss * weights

        if self.mean:
            return weighted_loss.mean()
        else:
            return weighted_loss.sum()


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
        vocab_size_encoder=None,
        vocab_size_decoder=None,
        embedding_share=True,
        embedding_sqrt_mul=True,
        abs_pos_encode=True,
        loss_mean=False
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
        self.embedding_share = embedding_share
        self.embedding_sqrt_mul = embedding_sqrt_mul
        self.abs_pos_encode = abs_pos_encode
        self.vocab_size_encoder = vocab_size_encoder
        self.vocab_size_decoder = vocab_size_decoder
        self.loss_mean = loss_mean


class TransformerTrainerConfig:
    def __init__(
        self,
        epochs=200,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        device='cpu',
        visdom_open=False,
        save=False
    ):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.optimizer = optimizer
        self.save = save
        self.visdom_open = visdom_open


class TransformerTrainer:
    def __init__(self, config, model, optimizer=torch.optim.Adam):
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def apply_init_weight(self, init_fn):
        self.model.apply(init_fn)

    def train(self, train_iter):
        model.to(device=self.config.device)
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.config.lr)
        window = None
        if self.config.visdom_open:
            window = Visdom()
            window.line([0], [0], win='loss', opts=dict(title='loss'))
        for i in range(self.config.epochs):
            loss_accumulated = 0
            tokens_accumulated = 0
            start = time.time()
            for batch in train_iter:
                en_in, en_len, de_en, de_len = [x.to(self.config.device) for x in batch]
                loss = model(en_in, en_len, de_en, de_len)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accumulated += loss.detach().cpu()
                tokens_accumulated += de_len.sum().detach().cpu()
            end = time.time()
            if self.config.visdom_open:
                window.line([loss_accumulated / tokens_accumulated],
                            [i],
                            win='loss',
                            opts=dict(title='loss'),
                            update='append'
                            )
            print(f'epoch: {i}, loss: {loss_accumulated / tokens_accumulated} per token')
            print(f'speed: {tokens_accumulated / (end - start)} tokens per sec on {self.config.device}')
        if self.config.save:
            torch.save(model.state_dict(), 'state_dic_transformer.pt')


if __name__ == '__main__':
    transformer_config = TransformerConfig()
    train_iter, src_vocab, tgt_vocab, source, target = tools.load_data_nmt(transformer_config.batch_size,
                                                                           transformer_config.sentence_max_len,
                                                                           num_examples=100
                                                                           )
    transformer_config.vocab_size_encoder = len(src_vocab)
    transformer_config.vocab_size_decoder = len(tgt_vocab)
    model = Transformer(transformer_config)
    trainer_config = TransformerTrainerConfig()
    trainer = TransformerTrainer(trainer_config, model)
    trainer.train(train_iter)
