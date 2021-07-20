# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from models.model_bases import summary
import torch
import os
import pickle
from collections import defaultdict
from models.conv_models import INFER, TRAIN
import logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key:
                str_losses.append("PPL({}) {:.3f}".format(key, avg_loss))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses)

    def total_loss(self):
        total = 0
        for key, loss in self.losses.items():
            if loss is None:
                continue
            total += np.average(loss)
        return total


def print_topic_words(decoder, vocab_dic, n_top_words=10, fname="none"):
    beta_exp = decoder.weight.data.cpu().numpy().T
    with open(fname, "w+", encoding="utf-8") as f:
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words-1:-1]]
            temp = sorted(beta_k)
            temp = [str(i) for i in temp]
            f.write(" ".join(temp)+"\n")
            f.write(" ".join([vocab_dic[w_id] for w_id in np.argsort(beta_k)])+"\n")
            yield 'Topic {}: {}'.format(k, ' '.join(x for x in topic_words))
        f.close()

def get_sent(model, data):
    sent = [model.vocab_bow[w_id] for w_id in data]
    return sent


def train(model, train_feed, valid_feed, test_feed, config, fold_num):
    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True) #says which message-positions to include in this batch
        stance = []
        true_stance = []
        while True:
            batch = train_feed.next_batch() #stances are in position here!
            if batch is None:
                break

            optimizer.zero_grad()
            loss, s, ts = model(batch) #mildly innacurate due to no id-checking but ok
            stance += s
            true_stance += ts
            if model.flush_valid:
                #logger.info("Flush previous valid loss")
                best_valid_loss = np.inf
                model.flush_valid = False
                optimizer = model.get_optimizer(config)

            model.backward(batch_cnt, loss)
            optimizer.step()
            batch_cnt += 1
            train_loss.add_loss(loss)

            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step,
                                                                         config.ckpt_step,
                                                                         model.kl_w)))
                # update l1 strength
                if config.use_l1_reg and batch_cnt <= config.freeze_step:
                    model.reg_l1_loss.update_l1_strength(model.ctx_decoder.weight)

            if batch_cnt % config.ckpt_step == 0:
                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

                #a = confusion_matrix(true_stance, stance)
                #print("STANCE:")
                #print("= Predicted C|D|Q|S =")
                #print(a)
                #print("Stance Macro F1:", round(f1_score(true_stance, stance, average="macro"), 3))

                #stance = []
                #true_stance = []
                #validation
                logging.info("Discourse Words:")
                logging.info('\n'.join(print_topic_words(model.x_decoder, model.vocab_bow, fname="w"+str(config.window_size)+"_x"+str(int(1/config.loss_mult))+"_ctx"+config.ctx_head+str(config.k)+"_tar"+config.tar_head+str(config.d)+"_disc"+str(fold_num)+".txt"))) #uncomment this line to write disc-decoder-weights to fike
                logging.info("Topic Words:")
                logging.info("\n".join(print_topic_words(model.ctx_decoder, model.vocab_bow, fname="w"+str(config.window_size)+"_x"+str(int(1/config.loss_mult))+"_ctx"+config.ctx_head+str(config.k)+"_tar"+config.tar_head+str(config.d)+"_topic"+str(fold_num)+".txt"))) #uncomment this line to write topic-decoder-weights to file
                #valid_loss = validate(model, valid_feed, config, batch_cnt)

                if done_epoch >= config.max_epoch:
                    return test2(model, valid_feed, config) #for getting F1 metrics from the left-out fold

                # exit eval model
                print("Epoch", done_epoch, "loss (before multiplication):", train_loss.total_loss())
                model.train()
                train_loss.clear()
                logger.info("\n**** Epcoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))

def test2(model, valid_feed, config):
    print("===FINAL RESULTS FOR THIS FOLD===")
    model.eval()
    valid_feed.epoch_init(config, shuffle=False, verbose=True, ignore_residual=False)
    stance = []
    true_stance = []
    veracity = []
    true_veracity = []
    identifiers = []
    thread_identifiers = []
    i = 0
    while True:
        i += 1
        if i%1000 == 0:
            print(">", i,"done")
        batch = valid_feed.next_batch()
        if batch is None:
            break
        _, s, ts = model(batch)
        stance += s
        true_stance += ts
    #done
    a = confusion_matrix(true_stance, stance)
    print("STANCE:")
    print("= Predicted C|D|Q|S =")
    print(a)
    print("Stance Macro F1:", round(f1_score(true_stance, stance, average="macro"), 4))
    #exit()

    return stance, true_stance, [], []


def inference(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()

    data_feed.epoch_init(config, ignore_residual=False, shuffle=num_batch is not None, verbose=False)

    logger.info("Inference: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    gen = []
    d_ids = []
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs = model(batch, mode=INFER)

        # move from GPU to CPU
        gen_ = outputs.gen.cpu().data.numpy()
        d_ids_ = outputs.d_ids.cpu().data.numpy()

        gen.append(gen_)
        d_ids.append(d_ids_)
    gen = np.concatenate(gen)
    # output discourse
    d_ids = np.concatenate(d_ids)
    rst = []

    for r_id, row in enumerate(data_feed.data):
        u_id = row.target.meta["id"]
        disc = row.target.meta["disc"]
        vec = gen[r_id]
        d_id = d_ids[r_id][0]
        rst.append({"id": u_id, "true_disc": disc, "pred_disc": d_id, "vec": vec})

    pickle.dump(rst, dest_f)
    logger.info("Inference Done")

