import copy
import torch
from visdom import Visdom
import time


class TransformerTrainerConfig:
    def __init__(
        self,
        epochs=200,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        device='cpu',
        visdom_open=False,
        save=False,
        loss_batch_mean=True
    ):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.optimizer = optimizer
        self.save = save
        self.visdom_open = visdom_open
        self.loss_batch_mean = loss_batch_mean


class GPT2TrainerConfig:
    def __init__(
        self,
        epochs=200,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        device='cpu',
        loss_batch_mean=True,
        visdom_open=False,
        save=False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.optimizer = optimizer
        self.loss_batch_mean = loss_batch_mean
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
        model = self.model
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
                if self.config.loss_batch_mean:
                    loss /= en_in.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accumulated += loss.detach().cpu()
                tokens_accumulated += de_len.sum().detach().cpu()
            end = time.time()
            if self.config.visdom_open:
                window.line([loss_accumulated / tokens_accumulated],
                            [i], win='loss',
                            opts=dict(title='loss'),
                            update='append'
                            )
            print(f'epoch: {i}, loss: {loss_accumulated / tokens_accumulated} per token')
            print(f'speed: {tokens_accumulated / (end - start)} tokens per sec on {self.config.device}')
        if self.config.save:
            torch.save(model.state_dict(), 'state_dic_transformer.pt')


class GPT2Trainer:
    def __init__(self, config, model, optimizer=torch.optim.Adam):
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def apply_init_weight(self, init_fn):
        self.model.apply(init_fn)

    def train(self, train_iter):
        model = self.model
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
                input_ids, input_mask = [x.to(self.config.device) for x in batch]
                # state = model.init_state()
                loss = model(input_ids, input_mask)['loss']
                if self.config.loss_batch_mean:
                    loss /= input_ids.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accumulated += loss.detach().cpu()
                tokens_accumulated += input_mask.sum().detach().cpu()
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


class GPT2RMTrainer:
    def __init__(self, config, model, optimizer=torch.optim.Adam):
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def apply_init_weight(self, init_fn):
        self.model.apply(init_fn)

    def train(self, train_iter):
        model = self.model
        model.to(device=self.config.device)
        model.train()
        optimizer = self.optimizer(model.parameters(), lr=self.config.lr)
        window = None
        if self.config.visdom_open:
            window = Visdom()
            window.line([0], [0], win='loss', opts=dict(title='loss'))
        for i in range(self.config.epochs):
            loss_accumulated = 0
            batch_num_accumulated = 0
            start = time.time()
            for batch in train_iter:
                batch_reward_table = torch.tensor([])

                # batch_ids sample like [[[1, 2, 3], [1, 2, 5]], [[4, 2, 3], [4, 2, 5]]]  batch_mask same
                batch_ids, batch_mask = [x.to(self.config.device) for x in batch]
                batch_size, sentence_pair_number, _ = batch_ids.shape
                # state = model.init_state()
                for sentence_idx in range(sentence_pair_number):
                    input_ids, input_mask = batch_ids[:, sentence_idx, :], batch_mask[:, sentence_idx, :]
                    reward = model(input_ids, input_mask)
                    batch_reward_table = torch.cat((batch_reward_table, reward), dim=-1)
                loss = model.rank_reward_loss(batch_reward_table)  # loss average on combinations not on batch
                if self.config.loss_batch_mean:
                    loss /= batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accumulated += loss.detach().cpu()
                batch_num_accumulated += batch_size
            end = time.time()
            if self.config.visdom_open:
                window.line([loss_accumulated / batch_num_accumulated],
                            [i],
                            win='loss',
                            opts=dict(title='loss'),
                            update='append'
                            )
            print(f'epoch: {i}, loss: {loss_accumulated / batch_num_accumulated} per sentence')
            print(f'speed: {batch_num_accumulated / (end - start + 1e-7)} sentences per sec on {self.config.device}')
        if self.config.save:
            torch.save(model.state_dict(), 'state_dic_transformer.pt')


class PPOTrainerConfig:
    def __init__(
        self,
        epochs=20,
        ppo_epochs=4,
        lr=1e-4,
        reward_discount=1,
        value_clip_range=0.2,
        prob_clip_range=0.2,
        entropy_penalty_sft=0.2,
        entropy_bonus_rl=0.2,
        value_fuc_coe=1,
        pretraining_coe=0,
        device='cpu',
        entropy_loss=False,
        loss_batch_mean=True,
        visdom_open=False,
        save=False,
        optimizer=torch.optim.Adam,
    ):
        self.epochs = epochs
        self.ppo_epochs = ppo_epochs
        self.lr = lr
        self.reward_discount = reward_discount
        self.value_clip_range = value_clip_range
        self.prob_clip_range = prob_clip_range
        self.entropy_penalty_sft = entropy_penalty_sft
        self.entropy_bonus_rl = entropy_bonus_rl
        self.value_fuc_coe = value_fuc_coe
        self.pretraining_coefficient = pretraining_coe
        self.device = device
        self.entropy_loss = entropy_loss
        self.loss_batch_mean = loss_batch_mean
        self.visdom_open = visdom_open
        self.save = save
        self.optimizer = optimizer


class PPOTrainer:
    """
    train a model with ppo
    """

    def __init__(self, config, model, vf_model):
        self.config = config
        self.pg_model = model
        self.refer_model = copy.deepcopy(model)
        self.vf_model = vf_model
        self.pg_optimizer = config.optimizer(self.pg_model.parameters())
        self.vf_optimizer = config.optimizer(self.vf_model.parameters())
        self.refer_model.eval()

    def set_eval(self):
        self.pg_model.eval()
        self.vf_model.eval()

    def set_train(self):
        self.pg_model.train()
        self.vf_model.train()

    def train(self, train_iter):
        """
        train model with ppo, only need prompt input,
            :param
                train_iter:  torch DataLoader
                             including sentence(concat [prompt, ans]), mask(prompt: 0 ans: 1), reward
                             batch_sentence shape: torch.size([batch, sentence_len])
                             batch_mask shape: torch.size([batch, sentence_len])
                             batch_reward shape: torch.size([batch])
            :return:
                training information
        """
        pg_model, refer_model, vf_model = self.pg_model, self.refer_model, self.vf_model
        config = self.config

        for i in range(config.epochs):

            pg_loss_accu = vf_loss_accu = entropy_loss_accu = 0
            batch_num_accu = 0
            token_num_accu = 0
            start = time.time()

            for batch in train_iter:
                # batch_ids shape: torch.size([batch, sentence_len])
                # batch_mask shape: torch.size([batch, sentence_len])
                # batch_reward shape: torch.size([batch])
                batch_ids, batch_mask, reward = [x.to(self.config.device) for x in batch]
                with torch.no_grad():
                    self.set_eval()
                    prob_refer, _ = self.prob_forward(refer_model, batch_ids)
                    prob_base, _ = self.prob_forward(pg_model, batch_ids)
                    value_base = self.value_forward(vf_model, batch_ids)

                episode_returns, advantages = self.returns_and_advantages(prob_refer,
                                                                          prob_base,
                                                                          value_base,
                                                                          reward,
                                                                          batch_mask
                                                                          )
                self.set_train()

                for ppo_epoch in range(self.config.ppo_epochs):
                    pro_update, logits_update = self.prob_forward(pg_model, batch_ids)
                    value_update = self.value_forward(vf_model, batch_ids)
                    pg_loss, vf_loss, entropy_loss = self.loss_compute(pro_update,
                                                                       prob_base,
                                                                       advantages,
                                                                       value_update,
                                                                       value_base,
                                                                       episode_returns,
                                                                       batch_mask,
                                                                       logits_update
                                                                       )

                    pg_loss_accu += pg_loss.detach()
                    vf_loss_accu += vf_loss.detach()
                    entropy_loss_accu += entropy_loss
                    batch_num_accu += batch_mask.shape[0]
                    token_num_accu += batch_mask[:, 1:].sum()

                    if self.config.entropy_loss:
                        pg_loss += entropy_loss
                    self.vf_optimizer.zero_grad()
                    self.pg_optimizer.zero_grad()
                    vf_loss.backward()
                    pg_loss.backward()
                    self.vf_optimizer.step()
                    self.pg_optimizer.step()

            end = time.time()

            ave_pg_loss = pg_loss_accu / batch_num_accu / self.config.ppo_epochs
            ave_vf_loss = vf_loss_accu / batch_num_accu / self.config.ppo_epochs
            ave_entropy_loss = entropy_loss_accu / batch_num_accu / self.config.ppo_epochs

            print(f'pg_loss: {ave_pg_loss} on epoch {i}')
            print(f'vf_loss: {ave_vf_loss} on epoch {i}')
            if self.config.entropy_loss:
                print(f'entropy_loss: {ave_entropy_loss} on epoch {i}')
            print(f'speed: {int(token_num_accu / (end - start + 1e-7))} tokens per sec on {self.config.device}')

        if self.config.save:
            torch.save(self.pg_model.state_dict(), 'state_dic_pg_model.pt')
            torch.save(self.vf_model.state_dict(), 'state_dic_vf_model.pt')

    def loss_compute(self,
                     pro_update,
                     prob_base,
                     advantages,
                     value_update,
                     value_base,
                     episode_returns,
                     batch_mask,
                     logits_update
                     ):

        weight_mask = batch_mask[:, 1:]
        value_update_clipped = torch.clamp(value_update,
                                           min=value_base - self.config.value_clip_range,
                                           max=value_base + self.config.value_clip_range
                                           )
        vf_obj = torch.max((value_update_clipped - episode_returns) ** 2,
                           (value_update - episode_returns) ** 2
                           )
        vf_loss = self.masked_loss(vf_obj, weight_mask)

        prob_ratio = pro_update / prob_base
        prob_ratio_clipped = torch.clamp(prob_ratio,
                                         1 - self.config.prob_clip_range,
                                         1 + self.config.prob_clip_range
                                         )
        surrogate_obj = torch.min(prob_ratio * advantages, prob_ratio_clipped * advantages)
        pg_loss = - self.masked_loss(surrogate_obj, weight_mask)

        entropy_loss = - self.masked_entropy_loss(logits_update, weight_mask)

        return pg_loss, vf_loss, entropy_loss

    @staticmethod
    def masked_loss(obj_fuc, weight_mask):
        """
        loss: mean on each token for each sentence
        """
        obj_fuc = (obj_fuc * weight_mask) / weight_mask.sum(dim=-1, keepdim=True)
        return obj_fuc.sum()

    @staticmethod
    def masked_entropy_loss(logits, weight_mask):
        """
        loss: mean on each token for each sentence
        """
        prob = torch.softmax(logits[:, :-1, :], dim=-1)
        entropy = - (prob * torch.log(prob)).sum(dim=-1)
        masked_entropy = entropy * weight_mask
        weighted_entropy = masked_entropy / weight_mask.sum(dim=-1, keepdim=True)
        return weighted_entropy.sum()

    @staticmethod
    def prob_forward(model, batch_ids):
        """
        gather the next token prob, exclude the first token
        """
        logits = model(batch_ids)['logits']
        prob = torch.softmax(logits[:, :-1, :], dim=-1)

        return torch.gather(prob, -1, batch_ids[:, 1:].unsqueeze(-1)).squeeze(-1), logits

    @staticmethod
    def value_forward(model, batch_ids):
        """
        return the state value of tokens, exclude the last token
        """
        return model.vf_forward(batch_ids)[:, :-1]

    def returns_and_advantages(self, prob_refer, prob_base, value_base, reward, batch_mask):
        """
        compute the advantages of the next token chose from the end of prompt
        compute the state values from the end of prompt, exclude the last token of answer
        """
        entropy_penalty_sft = torch.log(prob_refer)
        entropy_bonus_rl = -torch.log(prob_base)
        entropy_reward = entropy_penalty_sft + entropy_bonus_rl
        weight_mask = batch_mask[:, 1:]
        value_base = value_base * weight_mask
        entropy_reward = entropy_reward * weight_mask

        for i in range(entropy_reward.shape[0]):
            last_mask_idx = weight_mask[i].nonzero()[-1]
            entropy_reward[i][last_mask_idx] += reward[i]

        advantages = torch.zeros_like(entropy_reward)
        sentence_len = entropy_reward.shape[-1]
        advantages_accu = 0

        for t in reversed(range(sentence_len)):
            nx_val = value_base[:, t + 1] if t < sentence_len - 1 else 0
            delta = entropy_reward[:, t] + self.config.reward_discount * nx_val - value_base[:, t]
            advantages_accu = delta + self.config.reward_discount * advantages_accu
            advantages[:, t] = advantages_accu

        advantages = advantages * weight_mask
        episode_returns = advantages + value_base

        return episode_returns, advantages
