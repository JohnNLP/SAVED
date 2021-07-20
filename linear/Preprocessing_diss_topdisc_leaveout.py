import re
import math
import pickle
import numpy as np
import glob

# load in word list data
with open("./Models/lexicon/positive-words.txt") as f:
    pos = set(f.read().splitlines())
with open("./Models/lexicon/negative-words.txt") as f:
    neg = set(f.read().splitlines())
with open("./Models/lexicon/question-words.txt") as f:
    que = set(f.read().splitlines())

# load in GloVe data
with open("glove_50_max.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

# use GloVe to get word embeddings (sometimes replaced by embedding layer)
GLOVE_WORD_LIMIT = 200
EMBEDDING_DIM = 50


def glove(text):
    vec = np.zeros([GLOVE_WORD_LIMIT, EMBEDDING_DIM])
    for v, i in enumerate(text):
        if v >= GLOVE_WORD_LIMIT:
            break
        try:
            vec[v] = embeddings_dict[i]
        except:
            pass  # leave as 0s
    return vec


# hardcoded attention heuristic
def attention(text, vectors):
    text = text[:GLOVE_WORD_LIMIT]
    avg = np.zeros(EMBEDDING_DIM)
    if len(text) == 0:
        return avg.tolist()
    multiplier = 1
    counter = 0
    for v, i in enumerate(text):
        if i not in embeddings_dict:
            continue
        counter += 1
        # special words
        if i in pos or i in neg or i in que:
            multiplier = 1.5
        avg += multiplier * vectors[v]
        multiplier = 1
        # words after hashtag
        if i == "<hashtag>":
            multiplier = 1.5
    avg = avg / counter
    return avg.tolist()


# setup elmo
# print("Setting up ELMo...")
# from allennlp.commands.elmo import ElmoEmbedder
# weights = "./Models/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
# options = "./Models/ELMo/a.json"
# elmo = ElmoEmbedder(weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5", options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json")
# print("done")

# use elmo to get word embeddings (too slow!)
# def elmo_vec(text):
#    print("embedding...")
#    vectors = elmo.embed_sentence(text)
#    print(vectors[0])
#    print("===")
#    print(vectors[1])
#    return(vectors)


def top_disc(fname, t, d):
    print("Loading topic-disc...")

    disc_words = []
    file = glob.glob("*disc"+fname[0]+".txt")[0]
    print("File:",file)
    with open(file, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line[-1:] == "\n":
                line = line[:-1]
            line = line.split(" ")
            disc_words.append([])
            for word in line:
                disc_words[n].append(word)
    sanity_check = int(len(disc_words) / 2)
    if sanity_check != d:
        print("File has", sanity_check, "discourses but was expecting", d)
        exit()
    disc_w = []
    disc_n = []
    for v, i in enumerate(disc_words):
        if v % 2 == 0:
            disc_n.append(i)
        else:
            disc_w.append(i)

    for v, i in enumerate(disc_n):
        disc_n[v] = [float(j) for j in i]

    disc_all = []  # a list of <dimensions> dicts with each dict mapping dim_index+word+index : number
    for v, i in enumerate(disc_w):  # v = dimension index, i = words of vth dimention
        disc_all.append({})
        for w, j in enumerate(i):  # w = index of word in vth dimension, j = wth word of vth dimension
            number = disc_n[v][w]  # number = number for wth word of vth dimension as indexed in disc_n
            disc_all[v][j] = number

    topic_words = []
    file = glob.glob("*topic"+fname[0]+".txt")[0]
    print("File:",file)
    with open(file, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line[-1:] == "\n":
                line = line[:-1]
            line = line.split(" ")
            topic_words.append([])
            for word in line:
                topic_words[n].append(word)
    sanity_check = int(len(topic_words) / 2)
    if sanity_check != t:
        print("File has", sanity_check, "topics but was expecting", t)
        exit()
    topic_w = []
    topic_n = []
    for v, i in enumerate(topic_words):
        if v % 2 == 0:
            topic_n.append(i)
        else:
            topic_w.append(i)

    for v, i in enumerate(topic_n):
        topic_n[v] = [float(j) for j in i]

    topic_all = []
    for v, i in enumerate(topic_w):
        topic_all.append({})
        for w, j in enumerate(i):
            number = topic_n[v][w]
            topic_all[v][j] = number

    print("> done")

    return (disc_all, topic_all)
    # return(disc_all, topic_all)


def top_disc_2(fname):
    pass


# holds all the features of a tweet or reddit post
class Features:
    def __init__(self, t, d):
        self.TOPICS = t
        self.DISCOURSES = d

        self.topic = [0] * t  # todo: change
        self.discourse = [0] * d  # todo: change

        self.everything = None  # the data dict, for post-ptocessing
        self.text = None  # the text
        self.category = None  # which class is actually it in?
        self.pred_category = None  # which class is it predicted to be in?
        self.base_rumour = None  # the base rumour text
        self.vector = None  # weighted word vector average - rudimentary
        self.text_bert = None  # text for BERT

        self.caps = None  # (binary) has a reasonable number of CAPS?
        self.urls = None  # (binary) has URL?
        self.score = None  # number of upvotes or similar
        self.exclamations = None  # exclamation marks
        self.questions = None  # question marks
        self.source = None  # (binary) twitter = 1, reddit = 0

        self.direct_response = None  # is the response directly to the rumour?
        self.deleted = None  # is the response just "deleted" or "removed"?
        self.length = None  # log scale length of post
        self.blank = None  # for posts with no content
        self.quotes = None  # does the tweet contain quotes?
        self.pos_words = None  # agreeing words
        self.neg_words = None  # denying words
        self.que_words = None  # querying words
        self.hashtags = None  # at least 1 hashtag?
        self.mentions = None  # at least 2 mentions?
        self.user_verified = None  # is the author verified? #todo: twitter only? could set rep-overall thresh for reddit too?
        self.has_image = None  # does it have a picture?
        self.username = None  # who is the author?
        self.cred_true = None  # account's tendency of true
        self.cred_false = None  # account's tendency of false
        self.cred_unv = None  # account's tendency of unverified

        self.rep_time = None
        self.rep_follow = None
        self.rep_status = None
        self.rep_friends = None
        self.rep_geo = None
        self.rep_bg = None
        self.rep_profile = None

        self.rumour_class = None  # for base rumours only, is it asking or proposing?
        self.rumour_support = None  # supporting responses feature
        self.rumour_deny = None
        self.rumour_query = None
        self.rumour_comment = None
        self.rumour_aggregate = None  # weighted from the above 4 features
        self.rumour_news = None  # does it come from a news source?
        self.tree = None  # tree of all comments, base rumour only

        self.rumour_domain = None  # todo: (maybe) these features
        self.prev_class = None
        self.cosine = None

        self.d_rep = None
        self.d_oth = None
        self.t_rep = None
        self.t_oth = None
        self.d_count = None

    def __repr__(self):  # for neat printing
        return ("Class " + str(self.category) + " " + str(self.text))

    def debug(self):  # prints lots of things
        print("sou", self.source, "cat", self.category, "CAPS", self.caps, "URLs", self.urls, "score", self.score,
              "quotes", self.quotes, "excl", self.exclamations, "ques", self.questions, "text", self.text, "tree",
              self.tree, "vector", self.vector)


# given a string of text, preorocess it.
# does not hanble score or category.
def preprocess(text, disc_all, topic_all, topics, discourses):
    t = Features(topics, discourses)
    # deleted or removed
    t.deleted = 0
    if text == "[deleted]" or text == "[removed]":
        t.deleted = 1
    # blank
    t.blank = 0
    if text == "":
        t.blank = 1
    # detect and remove URLs (regex from my CS918 coursework)
    t.urls = len(re.findall(
        r"((https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*",
        text))
    if t.urls > 1:
        t.urls = 1
    text = re.sub(
        r"((https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*",
        "urltokenurlabc", text)
    # remove user names from user mentions
    text = re.sub(r"@[A-Za-z0-9_]+", "@", text)
    # quotes
    t.quotes = len(re.findall(r"\"", text))
    if t.quotes > 1:
        t.quotes = 1
    # exclamation marks
    t.exclamations = len(re.findall(r"!", text))
    if t.exclamations > 1:
        t.exclamations = 1
    # question marks
    t.questions = len(re.findall(r"\?", text))
    if t.questions > 1:
        t.questions = 1
    # caps
    t.caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    if t.caps > 1:
        t.caps = 1
    # remove undesirable characters
    text = re.sub(r"[^A-Za-z0-9\-_#@!?.,' ]", " ", text)
    # lowercasing
    text = text.lower()
    # expand apostrophes - works for everything except 's which aren't in GloVe anyway
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"there's", "there is", text)
    # bert split off
    t.text_bert = text
    # numbers
    text = re.sub(r"\b[0-9][0-9,]*(\.[0-9]+)?\b", " <number> ", text)
    # handle tokens (keeps punctuation)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"@", " <user> ", text)
    text = re.sub(r"#", " <hashtag> ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"urltokenurlabc", " <url> ", text)
    # get base class (rumours only)
    if "?" in text or "is it true" in text or "debunk this" in text:
        t.rumour_class = 1  # query
    else:
        t.rumour_class = 0  # support
    # tokenize
    text = text.split()
    t.length = min(5, math.log(len(text) + 1)) / 5
    # get word vectors
    temp_vector = glove(text)
    t.vector = attention(text, temp_vector)
    # hashtags
    t.hashtags = min(1, text.count("<hashtag>"))
    # mentions
    t.mentions = 0
    if text.count("<user>") > 1:
        t.mentions = 1
    # pos, neg, que words
    t.pos_words = len([i for i in text if i in pos]) / 4
    t.neg_words = len([i for i in text if i in neg]) / 4
    t.que_words = len([i for i in text if i in que]) / 4
    '''
    # discourses
    temp_count = 0
    for i in text:
        if i in disc_all[0]: #all discourses have the same words anyway -- for each word
            temp_count += 1
            for v, j in enumerate(disc_all): #for each dimension
                t.discourse[v] += disc_all[v][i] #add number to dimension as indexed by dim_number+word
    #print(t.discourse)
    #print(temp_count)
    temp = [abs(i) for i in t.discourse] #check tweet length is not 0
    highest = max(temp)
    if highest != 0:
        t.discourse = [i / temp_count for i in t.discourse]
    #print(t.discourse)
    '''

    # topics
    temp_count = 0
    for i in text:
        if i in topic_all[0]:  # all discourses have the same words anyway
            temp_count += 1
            for v, j in enumerate(topic_all):
                t.topic[v] += topic_all[v][i]
    # print(t.discourse)
    # print(temp_count)
    temp = [abs(i) for i in t.topic]
    highest = max(temp)
    if highest != 0:
        t.topic = [i / temp_count for i in t.topic]

    # discs
    temp_count = 0
    for i in text:
        if i in disc_all[0]:  # all discourses have the same words anyway
            temp_count += 1
            for v, j in enumerate(disc_all):
                t.discourse[v] += disc_all[v][i]
    # print(t.discourse)
    # print(temp_count)
    temp = [abs(i) for i in t.discourse]
    highest = max(temp)
    if highest != 0:
        t.discourse = [i / temp_count for i in t.discourse]

    t.text = text
    return t
