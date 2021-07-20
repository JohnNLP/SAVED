# -*- coding: utf-8 -*-
import itertools

import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.encoders import MultiFC
import criterions
from utils import FLOAT, LONG, cast_type
from models.model_bases import BaseModel
from utils import Pack
import numbers
import json

INFER = "infer"
TRAIN = "train"

class TDM(BaseModel):
    logger = logging.getLogger(__name__)
    def __init__(self, corpus, config):
        super(TDM, self).__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_bow = corpus.vocab_bow
        self.vocab_bow_stopwords = corpus.vocab_bow_stopwords
        self.vocab_size = len(self.vocab_bow)
        if not hasattr(config, "freeze_step"):
            config.freeze_step = 6000

        # build mode here
        # x is for discourse
        self.x_encoder = MultiFC(self.vocab_size, config.hidden_size, config.d,
                                 num_hidden_layers=1, short_cut=True)

        self.x_generator = MultiFC(config.d, config.d, config.d,
                                   num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(config.d, self.vocab_size, bias=False)

        # context encoder
        # ctx is for topic
        self.ctx_encoder = MultiFC(self.vocab_size, config.hidden_size, config.hidden_size,
                                   num_hidden_layers=1, short_cut=True)
        self.ctx_encoder2 = MultiFC(self.vocab_size, config.hidden_size, config.hidden_size,
                                   num_hidden_layers=1, short_cut=True)
        self.q_z_mu, self.q_z_logvar = nn.Linear(config.hidden_size, config.k), nn.Linear(config.hidden_size, config.k)
        self.q_z_mu2, self.q_z_logvar2 = nn.Linear(config.hidden_size, config.k), nn.Linear(config.hidden_size, config.k)

        self.q_d_mu, self.q_d_logvar = nn.Linear(config.hidden_size, config.d), nn.Linear(config.hidden_size, config.d)

        # cnn
        # self.ctx_encoder = CtxEncoder(config, utt_encoder=self.utt_encoder)
        self.ctx_generator = MultiFC(config.k, config.k, config.k, num_hidden_layers=0, short_cut=False)
        self.ctx_generator2 = MultiFC(config.k, config.k, config.k, num_hidden_layers=0, short_cut=False)

        # decoder
        self.ctx_dec_connector = nn.Linear(config.k, config.k, bias=True)
        self.x_dec_connector = nn.Linear(config.d, config.d, bias=True)
        self.ctx_decoder = nn.Linear(config.k, self.vocab_size)
        self.ctx_decoder2 = nn.Linear(config.k, self.vocab_size)

        self.decoder = nn.Linear(config.k + config.d, self.vocab_size, bias=False)

        # connector
        self.cat_connector = GumbelConnector()
        self.nll_loss = criterions.PPLLoss(self.config)
        self.nll_loss_filtered = criterions.PPLLoss(self.config, vocab=self.vocab_bow,
                                                    ignore_vocab=self.vocab_bow_stopwords)
        self.kl_loss = criterions.GaussianKLLoss()
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()
        self.reg_l1_loss = criterions.L1RegLoss(0.70)
        self.log_uniform_d = Variable(torch.log(torch.ones(1) / config.d))

        #custom
        self.stance_model = MultiFC(config.k, 12, 4, num_hidden_layers=1, short_cut=False)
        self.veracity_model = MultiFC(config.k, 12, 3, num_hidden_layers=1, short_cut=False)
        self.factor_dim_mismatch = False
        if config.k != config.d:
            self.factor_dim_mismatch = True
        self.stance_dim_fixer = nn.Linear(config.d, config.k)

        if self.use_gpu:
            self.log_uniform_d = self.log_uniform_d.cuda()

    def qdx_forward(self, tar_utts): #x - i
        #qd_logits = self.x_encoder(tar_utts).view(-1, self.config.d) #(32, 6)
        #qd_logits_multi = qd_logits.repeat(self.config.d_size, 1, 1) #(1, 32, 6)
        #sample_d_multi, d_ids_multi = self.cat_connector(qd_logits_multi, 1.0,
        #                                                 self.use_gpu, return_max_id=True)
        #sample_d = sample_d_multi.mean(0)
        #d_ids = d_ids_multi.view(self.config.d_size, -1).transpose(0, 1)

        #return Pack(qd_logits=qd_logits, sample_d=sample_d, d_ids=d_ids)
        tar_out = F.tanh(self.ctx_encoder(tar_utts)) #not-2 here = param sharing
        d_mu = self.q_d_mu(tar_out)
        d_logvar = self.q_d_logvar(tar_out)
        sample_d = self.reparameterize(d_mu, d_logvar)
        return Pack(sample_d=sample_d, d_mu=d_mu, d_logvar=d_logvar)

    def pxy_forward(self, results): #x - ii
        gen_d = self.x_generator(results.sample_d)
        x_out = self.x_decoder(gen_d)

        results['gen_d'] = gen_d
        results['x_out'] = x_out

        return results

    def qzc_forward(self, ctx_utts): #c - i
        ctx_out = F.tanh(self.ctx_encoder(ctx_utts)) #(32, 18007)
        z_mu = self.q_z_mu(ctx_out)
        z_logvar = self.q_z_logvar(ctx_out)
        sample_z = self.reparameterize(z_mu, z_logvar)
        return Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def pcz_forward(self, results): #c - ii
        gen_c = self.ctx_generator(results.sample_z)
        c_out = self.ctx_decoder(gen_c)

        results['gen_c'] = gen_c
        results['c_out'] = c_out

        return results

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def valid_loss(self, loss, batch_cnt=None):
        vae_x_loss = loss.vae_x_nll + loss.vae_x_kl
        vae_c_loss = loss.vae_c_nll + loss.vae_c_kl
        div_kl = loss.div_kl
        dec_loss = loss.nll
        if "stance_loss_c" in loss and not "stance_loss_x" in loss:
            stance_loss = loss.stance_loss_c
        if "stance_loss_x" in loss and not "stance_loss_c" in loss:
            stance_loss = loss.stance_loss_x
        if "stance_loss_c" in loss and "stance_loss_x" in loss:
            stance_loss = loss.stance_loss_c + loss.stance_loss_x
        if "veracity_loss_c" in loss:
            veracity_loss = loss.veracity_loss_c
        if self.config.use_l1_reg:
            vae_c_loss += loss.l1_reg

        if batch_cnt is not None and batch_cnt > self.config.freeze_step:
            total_loss = dec_loss
            #total_loss = vae_x_loss
            self.flush_valid = True
            for param in self.x_encoder.parameters():
                param.requires_grad = False
            for param in self.x_generator.parameters():
                param.requires_grad = False
            for param in self.x_decoder.parameters():
                param.requires_grad = False
            for param in self.ctx_encoder.parameters():
                param.requires_grad = False
            for param in self.ctx_generator.parameters():
                param.requires_grad = False
            for param in self.ctx_decoder.parameters():
                param.requires_grad = False
            for param in self.stance_model.parameters():
                param.requires_grad = False
            for param in self.veracity_model.parameters():
                param.requires_grad = False
        else:
            #beta_exp = self.x_decoder.weight.data.cpu().numpy().T
            #for k, beta_k in enumerate(beta_exp):
            #    topic_words = [self.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-10 - 1:-1]]
            #    print(topic_words)
            mult1 = self.config.loss_mult
            if ("stance_loss_x" in loss or "stance_loss_c" in loss) and not "veracity_loss_c" in loss:
                total_loss = mult1*vae_x_loss + mult1*vae_c_loss + mult1*0.001*div_kl + stance_loss
            elif "veracity_loss_c" in loss and not ("stance_loss_x" in loss or "stance_loss_c" in loss):
                total_loss = mult1*vae_x_loss + mult1*vae_c_loss + mult1*0.001*div_kl + veracity_loss
            elif ("stance_loss_x" in loss or "stance_loss_c" in loss) and "veracity_loss_c" in loss:
                total_loss = mult1*vae_x_loss + mult1*vae_c_loss + mult1*0.001*div_kl + stance_loss + veracity_loss
            else:
                total_loss = mult1*vae_x_loss + mult1*vae_c_loss + mult1*0.001*div_kl
            #print(total_loss)
        return total_loss

    def forward(self, data_feed, mode=TRAIN, return_latent=False, save_discourse=False): #stance is in the data_feed!

        batch_size = len(data_feed['targets'])
        veracity_weights = data_feed["veracity_weights"][0]
        veracities = data_feed["thread_veracity"]
        stance_weights = data_feed["stance_weights"][0]
        stances = data_feed["stances"]

        ctx_utts = self.np2var(data_feed['contexts'], FLOAT)
        tar_utts = self.np2var(data_feed['targets'], FLOAT)

        vae_x_resp = self.pxy_forward(self.qdx_forward(tar_utts))

        ctx_utts = ctx_utts.sum(1)  # merge contexts into one bow
        vae_c_resp = self.pcz_forward(self.qzc_forward(ctx_utts))

        if save_discourse == True:
            metas = data_feed["thread_ids"]
            temp = vae_c_resp.sample_z #switch c_z is topic x_d is discourse
            temp = temp.cpu().tolist()
            json_dict = {}
            for v, i in enumerate(temp):
                json_dict[metas[v]] = i

        TAR_HEAD = self.config.tar_head
        CTX_HEAD = self.config.ctx_head

        do_stances = False

        stance_loss_c = None
        stance_loss_x = None
        predicted_stances = []
        condensed_stances = []
        condensed_ids = []
        if TAR_HEAD == "stance_pred" or TAR_HEAD == "stance_adv" or CTX_HEAD == "stance_pred" or CTX_HEAD == "stance_adv":
            temp_d = vae_x_resp.sample_d
            temp_z = vae_c_resp.sample_z
            temp_ids = data_feed["metas"]
            condensed_d = []
            condensed_z = []
            condensed_stances = []
            for v, i in enumerate(stances):
                if i is not None:
                    condensed_d.append(temp_d[v])
                    condensed_z.append(temp_z[v])
                    condensed_ids.append(temp_ids[v])
                    condensed_stances.append(i)
            if len(condensed_stances) > 0:
                do_stances = True
                condensed_d = torch.stack(condensed_d).to(self.device)
                condensed_z = torch.stack(condensed_z).to(self.device)
                condensed_stances = torch.tensor(condensed_stances).to(self.device)

                ctx_stances = self.stance_model(condensed_z)
                if self.factor_dim_mismatch:
                    tar_stances = self.stance_model(self.stance_dim_fixer(condensed_d))
                else:
                    tar_stances = self.stance_model(condensed_d)  # if sizes not equal then add an equalizing layer
                stance_weights = torch.tensor(stance_weights).to(self.device)  # move stuff over

                if TAR_HEAD == "stance_pred":
                    stance_loss_x = F.cross_entropy(tar_stances, condensed_stances, stance_weights)
                    tar_stances = torch.max(tar_stances, 1)[1]
                    predicted_stances = tar_stances.cpu().tolist()
                    condensed_stances = condensed_stances.cpu().tolist()
                elif TAR_HEAD == "stance_adv":
                    uniform_dist = torch.Tensor(len(tar_stances), 4).fill_((0.25)).to(self.device)  # uniform dist
                    stance_loss_x = F.kl_div(F.log_softmax(tar_stances), uniform_dist, reduction="sum") * 4
                if CTX_HEAD == "stance_pred":
                    stance_loss_c = F.cross_entropy(ctx_stances, condensed_stances, stance_weights)
                    ctx_stances = torch.max(ctx_stances, 1)[1]  # get confusion-matrix-friendly output
                    predicted_stances = ctx_stances.cpu().tolist()  # shunt everything back to cpu
                    condensed_stances = condensed_stances.cpu().tolist()
                elif CTX_HEAD == "stance_adv":
                    uniform_dist = torch.Tensor(len(ctx_stances), 4).fill_((0.25)).to(self.device)
                    stance_loss_c = F.kl_div(F.log_softmax(ctx_stances), uniform_dist, reduction="sum") * 4

        do_veracities = False

        # veracity stuff
        veracity_loss_c = None
        predicted_vercities = []
        condensed_veracities = []
        condensed_thread_ids = []
        if TAR_HEAD == "veracity_pred" or TAR_HEAD == "veracity_adv" or CTX_HEAD == "veracity_pred" or CTX_HEAD == "veracity_adv":
            temp_z = vae_c_resp.sample_z
            condensed_z = []
            condensed_veracities = []
            temp_thread_ids = data_feed["thread_ids"]
            for v, i in enumerate(veracities):
                if i is not None:
                    condensed_z.append(temp_z[v])
                    condensed_thread_ids.append(temp_thread_ids[v])
                    condensed_veracities.append(i)
            if len(condensed_veracities) > 0:
                do_veracities = True
                condensed_z = torch.stack(condensed_z).to(self.device)
                condensed_veracities = torch.tensor(condensed_veracities).to(self.device)

                ctx_veracities = self.veracity_model(condensed_z)
                veracity_weights = torch.tensor(veracity_weights).to(self.device)  # move stuff over

                if TAR_HEAD == "veracity_pred":
                    print("Makes no sense to predict veracity using a single message")
                    exit()
                elif TAR_HEAD == "veracity_adv":
                    print("Makes no sense to predict uniform veracity using a single message")
                    exit()
                if CTX_HEAD == "veracity_pred":
                    veracity_loss_c = F.cross_entropy(ctx_veracities, condensed_veracities, veracity_weights)
                    ctx_veracities = torch.max(ctx_veracities, 1)[1]  # get confusion-matrix-friendly output
                    predicted_vercities = ctx_veracities.cpu().tolist()  # shunt everything back to cpu
                    condensed_veracities = condensed_veracities.cpu().tolist()
                elif CTX_HEAD == "veracity_adv":
                    uniform_dist = torch.Tensor(len(ctx_veracities), 3).fill_((0.33)).to(self.device)
                    veracity_loss_c = F.kl_div(F.log_softmax(ctx_veracities), uniform_dist, reduction="sum") * 3

        # prior network (we can restrict the prior to stopwords and emotional words)
        # combine context topic and x discourse
        sample_z = vae_c_resp.sample_z.detach()
        sample_d = vae_x_resp.sample_d.detach()
        gen = torch.cat([self.x_dec_connector(sample_d), self.ctx_dec_connector(sample_z)], dim=1)
        dec_out = self.decoder(gen)

        # compute loss or return results
        if mode == INFER:
            print("INFER MODE, not sure what to do without d_ids")
            exit()

        log_qx = F.log_softmax(vae_x_resp.x_out, dim=1)
        vae_x_nll = self.nll_loss_filtered(log_qx, tar_utts, batch_size, unit_average=True)  # loss
        vae_x_kl = self.kl_loss(vae_x_resp.d_mu, vae_x_resp.d_logvar, batch_size, unit_average=True)

        log_qc = F.log_softmax(vae_c_resp.c_out, dim=1)
        vae_c_nll = self.nll_loss_filtered(log_qc, ctx_utts, batch_size, unit_average=True) #loss
        vae_c_kl = self.kl_loss(vae_c_resp.z_mu, vae_c_resp.z_logvar, batch_size, unit_average=True)

        div_kl = - self.cat_kl_loss(log_qx, log_qc, batch_size, unit_average=True)  # maximize the kl loss

        # decoder loss
        log_dec = F.log_softmax(dec_out, dim=1)
        dec_nll = self.nll_loss(log_dec, tar_utts, batch_size, unit_average=True) #todo: use of tar_utts is dubious

        # regularization loss
        if self.config.use_l1_reg:
            l1_reg = self.reg_l1_loss(self.ctx_decoder.weight, torch.zeros_like(self.ctx_decoder.weight))
        else:
            l1_reg = None

        if do_stances and not do_veracities:
            if stance_loss_x is None:
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                               vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_c=stance_loss_c)
            elif stance_loss_c is None:
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                               vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_x=stance_loss_x)
            else: #can return all
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                           vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_x=stance_loss_x, stance_loss_c=stance_loss_c)
        elif do_veracities and not do_stances:
            results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                       vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, veracity_loss_c=veracity_loss_c)
        elif do_stances and do_veracities:
            if stance_loss_x is None:
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                               vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_c=stance_loss_c, veracity_loss_c=veracity_loss_c)
            elif stance_loss_c is None:
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                               vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_x=stance_loss_x, veracity_loss_c=veracity_loss_c)
            else:
                results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                           vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl, stance_loss_x=stance_loss_x, stance_loss_c=stance_loss_c, veracity_loss_c=veracity_loss_c)
        else:
            results = Pack(nll=dec_nll, vae_x_nll=vae_x_nll, vae_x_kl=vae_x_kl, vae_c_nll=vae_c_nll,
                           vae_c_kl=vae_c_kl, l1_reg=l1_reg, div_kl=div_kl)
	
        if return_latent:
            results['gen'] = gen
            print("No d_ids")
            exit()
            #results['d_ids'] = d_ids

        #if save_discourse:
        #    return results, json_dict
        #else:
        return results, predicted_vercities, condensed_veracities

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)


class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, use_gpu, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, use_gpu)
        _, y_hard = torch.max(y, dim=-1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, use_gpu)
            y_onehot.scatter_(-1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y
