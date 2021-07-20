# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import logging
import os

import torch

import engine
from dataset import corpora
from dataset import data_loaders
from models import conv_models
from utils import str2bool, prepare_dirs_loggers, get_time

import gen_utils

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


data_arg = add_argument_group('Data')
net_arg = add_argument_group('Network')
train_arg = add_argument_group('Training')
misc_arg = add_argument_group('Misc')    

# Data
data_arg.add_argument('--data_dir', type=list, default=["data/output.json"])
    #'data/veracity_en_c.json',
    #"data/veracity_en_f.json",
    #"data/veracity_en_g.json",
    #"data/veracity_en_o.json",
    #"data/veracity_en_s.json"])
data_arg.add_argument('--log_dir', type=str, default='logs')

# commonly changed
net_arg.add_argument('--d', type=int, default=6) #default 50
net_arg.add_argument('--k', type=int, default=10) #default 50

train_arg.add_argument('--window_size', type=int, default=20) #default 20
train_arg.add_argument('--max_epoch', type=int, default=25) #default 100
train_arg.add_argument('--ctx_head', type=str, default="none") #stance_pred / stance_adv / veracity_pred / veracity_adv
train_arg.add_argument('--tar_head', type=str, default="none") #stance_pred / stance_adv
net_arg.add_argument('--loss_mult', type=float, default=1/200) #todo: go back to 1/200

#data_arg.add_argument('--load_sess', type=str, default="t50_d50_w20")

# Network
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--hidden_size', type=int, default=500)
net_arg.add_argument('--max_vocab_cnt', type=int, default=50000)

# Training / test parameters
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--step_size', type=int, default=1)
train_arg.add_argument('--init_w', type=float, default=0.1)
train_arg.add_argument('--init_lr', type=float, default=0.0003) #0.001 is default
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--use_l1_reg', type=str2bool, default=False)
train_arg.add_argument('--improve_threshold', type=float, default=0.996)
train_arg.add_argument('--patient_increase', type=float, default=4.0)
train_arg.add_argument('--early_stop', type=str2bool, default=False)

# MISC
misc_arg.add_argument('--output_vis', type=str2bool, default=False)
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--print_step', type=int, default=100)
misc_arg.add_argument('--ckpt_step', type=int, default=1000)
misc_arg.add_argument('--freeze_step', type=int, default=14500)
misc_arg.add_argument('--batch_size', type=int, default=32)
data_arg.add_argument('--token', type=str, default="")
logger = logging.getLogger()

def main(config):
    #print(config.k)
    #exit()
    config.session_dir = os.path.join(config.log_dir, "t"+str(config.k)+"_d"+str(config.d)+"_w"+str(config.window_size))
    try:
        os.mkdir(config.session_dir)
    except:
        pass
    stance = []
    true_stance = []
    print("TRAINING (NOT FINETUNING)")
    print("Events: " + str(len(config.data_dir)))
    print("t" + str(config.k) + " d" + str(config.d) + " w" + str(config.window_size) + " e" + str(config.max_epoch))
    prepare_dirs_loggers(config, os.path.basename(__file__))
    for i in range(len(config.data_dir)):
        corpus_client = corpora.TwitterCorpus(config, i)


        conv_corpus = corpus_client.get_corpus_bow() #stance is included here!
        train_conv, valid_conv, test_conv, vocab_size = conv_corpus['train'],\
                                            conv_corpus['valid'],\
                                            conv_corpus['test'],\
                                            conv_corpus['vocab_size']

        # create data loader that feed the deep models
        train_feed = data_loaders.TCDataLoader("Train", train_conv, vocab_size, config) #stance should be loaded in correctly
        valid_feed = data_loaders.TCDataLoader("Valid", valid_conv, vocab_size, config)
        test_feed = data_loaders.TCDataLoader("Test", test_conv, vocab_size, config)

        # for generation
        conv_corpus_seq = corpus_client.get_corpus_seq()
        train_conv_seq, valid_conv_seq, test_conv_seq = conv_corpus_seq['train'], conv_corpus_seq['valid'], conv_corpus_seq['test'] #stance is here too in case needed, but i doubt it

        model = conv_models.TDM(corpus_client, config)

        if config.use_gpu:
            model.cuda()

        s, ts, v, tv = engine.train(model, train_feed, valid_feed, test_feed, config, config.data_dir[i][-6])
        stance += s
        true_stance += ts

        '''
        # config.batch_size = 10
        train_feed_output = data_loaders.TCDataLoader("Train_Output", train_conv, vocab_size, config)
        test_feed_output = data_loaders.TCDataLoader("Test_Output", test_conv, vocab_size, config)
        valid_feed_output = data_loaders.TCDataLoader("Valid_Output", valid_conv, vocab_size, config)
    
    
        if config.output_vis:
            with open(os.path.join(config.session_dir, "gen_samples.txt"), "w") as gen_f:
                gen_utils.generate(model, valid_feed_output, valid_conv_seq, config, num_batch=2, dest_f=gen_f)
        '''

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    a = confusion_matrix(true_stance, stance)
    print("FINAL RESULTS:")
    print("= Predicted F|T|U =")
    print(a)
    print("Stance Macro F1:", round(f1_score(true_stance, stance, average="macro"), 4))

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
