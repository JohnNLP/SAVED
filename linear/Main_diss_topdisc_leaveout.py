import os
import json
from functools import reduce
from Preprocessing_diss_topdisc_leaveout import preprocess
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import random

TOPICS = 10
DISCS = 6
USING = "topic" #topic / discourse / both
print("T="+str(TOPICS)+" D="+str(DISCS)+" Using "+USING)

#################################################
################# DATA HANDLING #################
#################################################

print("File directory mapping...")

#base data directory
rootdir = "./Data/"

#build a tree of the data directories
#code by Andrew Clark from http://code.activestate.com/recipes/577879-create-a-nested-dictionary-from-oswalk/
dir = {} #soon-to-be tree of directories
rootdir = rootdir.rstrip(os.sep)
start = rootdir.rfind(os.sep) + 1
for path, dirs, files in os.walk(rootdir):
    folders = path[start:].split(os.sep)
    subdir = dict.fromkeys(files)
    parent = reduce(dict.get, folders[:-1], dir)
    parent[folders[-1]] = subdir
#end
print("> done")

#setting up paths to reddit/twitter train/dev/test directories
#absolute
train_dir = dir["./Data/all-rnr-annotated-threads"] #works on windows 10, otherwise try the below
#train_dir = dir["./Data/"]["all-rnr-annotated-threads"]
#train
train_dir_twitter = []
for i in train_dir: #inconsistency between reddit and twitter directories
    train_dir_twitter.append((i, train_dir[i]["rumours"])) #rumours only

#given a twitter directory, extract all the rumours and
#responses, preprocessing and putting them nicely into
#dictionaries.
def get_rumours_twitter(files, basepath,a,b):
    rumour_dict = {}
    for rumour in files: #iterate through the rumour+response files
        rumour_dict[rumour] = {"replies": {}}
        #carry_over_disc = []
        disc_count = 3
        with open(basepath+"/"+rumour+"/source-tweets/"+rumour+".json", encoding="utf-8") as f: #the base rumour
            data = json.load(f)
            all = data["text"]
            if data["lang"] != "en":
                continue
            score = data["favorite_count"] + data["retweet_count"]
            all = preprocess(all,a,b,TOPICS,DISCS)
            #carry_over_disc = [i*3 for i in all.discourse]
            all.everything = data
            all.score = score
            all.source = 1 #(binary) 1 = twitter
            all.username = data["user"]["name"]
            all.has_image = 0
            if "media" in data["entities"]:
                all.has_image = 1
            all.user_verified = 0
            if data["user"]["verified"] == True:
                all.user_verified = 1

            # true - misinformation = 0 true = 1
            # false - misinformation = 1 true = 0
            # unverified - misinformation = 0 true = 0
            try: #try to load annotation
                with open(basepath + "/" + rumour + "/annotation.json", encoding="utf-8") as g:  # response tree
                    info = json.load(g)
                    temp_true = int(info["true"])
                    temp_misinformation = int(info["misinformation"])
                    if temp_true == 1 and temp_misinformation == 0:
                        all.category = "true"
                        #print("true")
                    elif temp_true == 0 and temp_misinformation == 1:
                        all.category = "false"
                        #print("false")
                    elif temp_true == 0 and temp_misinformation == 0:
                        all.category = "unverified"
                        #print("unverified")
                    else:
                        print("strange truth value found", temp_true, temp_misinformation)
            except: #not all rumours have a class provided
                rumour_dict.pop(rumour)
                print("Error reading truth value")
                continue
            with open (basepath+"/"+rumour+"/structure.json", encoding="utf-8") as g: #response tree
                tree = json.load(g)
            all.tree = tree
            rumour_dict[rumour]["rumour"] = all
        disc_count = 1
        disc_imm = [0]*DISCS #todo: maybe can do better than 0s
        disc_oth = [0]*DISCS
        t_imm = [0]*TOPICS
        t_oth = [0]*TOPICS
        for r in files[rumour]["reactions"]:
            with open(basepath+"/"+rumour+"/reactions/"+r) as f: #responses to base rumour
                data = json.load(f)
                tag = r[:-5] #remove the ".json" to get response id
                response = data["text"]
                if data["lang"] != "en":
                    continue
                score = data["favorite_count"] + data["retweet_count"]
                response = preprocess(response,a,b,TOPICS,DISCS) #preprocessing
                #postprocessing below here
                response.everything = data
                response.score = score
                response.source = 1
                response.username = data["user"]["name"]
                response.has_image = 0
                if "media" in data["entities"]:
                    response.has_image = 1
                response.user_verified = 0
                if data["user"]["verified"] == True:
                    response.user_verified = 1
                response.direct_response = 0
                if tag in rumour_dict[rumour]["rumour"].tree[rumour]:
                    response.direct_response = 1
                rumour_dict[rumour]["replies"][tag] = response

                if response.direct_response == 1:
                    disc_count += 1
                    disc_imm = [a + b for a, b in zip(disc_imm, response.discourse)]
                    t_imm = [a + b for a, b in zip(t_imm, response.topic)]
                else:
                    disc_count += 1
                    disc_oth = [a + b for a, b in zip(disc_oth, response.discourse)]
                    t_oth = [a + b for a, b in zip(t_oth, response.topic)]
                #multiplier = 1+response.direct_response
                #carry_over_disc = [i+response.discourse[v]*multiplier for v, i in enumerate(carry_over_disc)]
                #disc_count += multiplier

            #rumour_dict[rumour]["rumour"].discourse = [i/multiplier for i in carry_over_disc]
            rumour_dict[rumour]["rumour"].d_rep = disc_imm
            rumour_dict[rumour]["rumour"].d_oth = disc_oth
            rumour_dict[rumour]["rumour"].t_rep = t_imm
            rumour_dict[rumour]["rumour"].t_oth = t_oth
            rumour_dict[rumour]["rumour"].d_count = disc_count
    return rumour_dict

#####################################################
############## CLASS INDICES FUNCTIONS ##############
#####################################################

#these are always ordered alphabetically

def makeClassIndices(data):
    temp = []
    for i in data:
        #TASK A-1
        if i == "comment":
            temp.append(0)
        elif i == "other":
            temp.append(1)
        #TASK A-2
        elif i == "deny":
            temp.append(0)
        elif i == "query":
            temp.append(1)
        elif i == "support":
            temp.append(2)
        #TASK B
        elif i == "false":
            temp.append(0)
        elif i == "true":
            temp.append(1)
        elif i == "unverified":
            temp.append(2)
    return temp

#TASK B
def removeClassIndicesB(data):
    temp = []
    for i in data:
        if i == 0:
            temp.append("false")
        elif i == 1:
            temp.append("true")
        elif i == 2:
            temp.append("unverified")
    return temp

#####################################################
####################### LOOCV #######################
#####################################################

all_events = []
for i in train_dir_twitter:
    all_events.append(i[0])

all_actual = []
all_pred = []

for num, cv_event in enumerate(all_events):
    if num < 0:
        continue

    from Preprocessing_diss_topdisc_leaveout import top_disc
    a,b = top_disc(cv_event, TOPICS, DISCS)
    print("disc size:", len(b[0]))

    # setup the training data dicts
    print("Setting up data...")
    all_tweets = {}
    for i in train_dir_twitter:
        print("> processing", i[0][:-16])
        temp = get_rumours_twitter(i[1], "./Data/all-rnr-annotated-threads/" + i[0] + "/rumours",a,b)
        all_tweets[i[0]] = temp  # stratify by event

    test_split = [cv_event]
    print("EVENT:",cv_event)
    train_split = list(all_tweets.keys())
    train_split.remove(cv_event)

    train_twitter = {}
    for i in train_split:
        train_twitter.update(all_tweets[i])
    test_twitter = {}
    for i in test_split:
        test_twitter.update(all_tweets[i])

    ####################################################
    #################### MODELLING #####################
    ####################################################

    print("MODELLING...")

    def makeTrainableTaskB(dict1=None, dict2=None, strip=False):
        x_d = []
        x_t = []
        x_features = []
        x_count = []
        y = []
        tags = []
        if dict2 != None:
            dict1.update(dict2)
        for r in dict1:
            t = dict1[r]["rumour"]
            d_rep = t.d_rep
            if d_rep == None:
                d_rep = [0]*DISCS #todo: maybe can do better than 0s
            d_oth = t.d_oth
            if d_oth == None:
                d_oth = [0]*DISCS
            t_rep = t.t_rep
            if t_rep == None:
                t_rep = [0]*TOPICS
            t_oth = t.t_oth
            if t_oth == None:
                t_oth = [0]*TOPICS
            count = t.d_count
            if count == None:
                count = 1
            if (count < 10 and strip):
                continue
            y.append(t.category)
            x_d.append([t.discourse,d_rep,d_oth])
            x_t.append([t.topic,t_rep,t_oth])
            x_features.append([t.user_verified])
            x_count.append(count)
            tags.append(r)
        return x_features, x_d, x_t, y, tags, x_count

    train_xB_f, train_xB_d, train_xB_t, train_yB, train_tagsB, train_count = makeTrainableTaskB(train_twitter.copy())
    test_xB_f, test_xB_d, test_xB_t, test_yB, test_tagsB, test_count = makeTrainableTaskB(test_twitter.copy(), strip=True)

    # shuffle
    temp = list(zip(train_xB_f, train_xB_d, train_xB_t, train_yB, train_count))
    random.shuffle(temp)
    train_xB_f, train_xB_d, train_xB_t, train_yB, train_count = zip(*temp)

    # Pytorch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class NeuralNetB(nn.Module):
        def __init__(self):
            super(NeuralNetB, self).__init__()
            self.mult_d = nn.Parameter(torch.rand(3, device=device))
            self.mult_t = nn.Parameter(torch.rand(3, device=device))
            if USING == "topic":
                self.fc1 = nn.Linear(TOPICS+1, 400)
            elif USING == "discourse":
                self.fc1 = nn.Linear(DISCS+1, 400)
            elif USING == "both":
                self.fc1 = nn.Linear(TOPICS+DISCS+1, 400)
            else:
                print("Wrong parameter set for topic/discourse/both")
                exit()
            self.relu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.2)
            self.fc2 = nn.Linear(400, 50)
            self.fc3 = nn.Linear(50, 3)

        def forward(self, x_f, x_d, x_t, tc):
            if USING == "topic":
                x_t2 = x_t/tc.view(tc.shape[0],1,1) #get partial averages
                x_t2 = x_t2*(self.mult_t*10).view(1,3,1) #multiply by weights
                out = torch.sum(x_t2, dim=1) #sum them up
            elif USING == "discourse":
                x_d2 = x_d/tc.view(tc.shape[0],1,1) #get partial averages
                x_d2 = x_d2*(self.mult_d*10).view(1,3,1) #multiply by weights
                out = torch.sum(x_d2, dim=1) #sum them up
            elif USING == "both":
                x_t2 = x_t / tc.view(tc.shape[0], 1, 1)  # get partial averages
                x_t2 = x_t2 * (self.mult_t * 10).view(1, 3, 1)  # multiply by weights
                out_t = torch.sum(x_t2, dim=1)  # sum them up
                x_d2 = x_d / tc.view(tc.shape[0], 1, 1)  # get partial averages
                x_d2 = x_d2 * (self.mult_d * 10).view(1, 3, 1)  # multiply by weights
                out_d = torch.sum(x_d2, dim=1)  # sum them up
                out = torch.cat((out_d, out_t),1)
            else:
                print("Wrong parameter set for topic/discourse/both")
                exit()

            out = torch.cat((out,x_f),1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.fc3(out)
            return out

    model = NeuralNetB().to(device)

    # account for class imbalance
    a = train_yB.count("false")
    b = train_yB.count("true")
    c = train_yB.count("unverified")
    weights = torch.tensor([1 / a, 1 / b, 1 / c], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    score_list = []
    predB = []

    # make tensors and move them to gpu
    train_xB_f = torch.tensor(train_xB_f, dtype=torch.float).to(device)
    test_xB_f = torch.tensor(test_xB_f, dtype=torch.float).to(device)
    train_count = torch.tensor(train_count, dtype=torch.float).to(device)
    test_count = torch.tensor(test_count, dtype=torch.float).to(device)
    train_xB_d = torch.tensor(train_xB_d, dtype=torch.float).to(device)
    test_xB_d = torch.tensor(test_xB_d, dtype=torch.float).to(device)
    train_xB_t = torch.tensor(train_xB_t, dtype=torch.float).to(device)
    test_xB_t = torch.tensor(test_xB_t, dtype=torch.float).to(device)
    train_yB = makeClassIndices(train_yB)
    train_yB = torch.tensor(train_yB, dtype=torch.long).to(device)

    # train model
    BATCH_SIZE = 12
    train_xB_f = torch.split(train_xB_f, BATCH_SIZE)
    train_xB_d = torch.split(train_xB_d, BATCH_SIZE)
    train_xB_t = torch.split(train_xB_t, BATCH_SIZE)
    train_count = torch.split(train_count, BATCH_SIZE)
    train_yB = torch.split(train_yB, BATCH_SIZE)
    results = []
    for epoch in range(200):
        model.train()
        loss_readout = 0
        for v, i in enumerate(train_xB_d):
            y_pred = model(train_xB_f[v], train_xB_d[v], train_xB_t[v], train_count[v])
            model.zero_grad()
            loss = criterion(y_pred, train_yB[v])
            loss.backward()
            optimizer.step()
            loss_readout += loss.item()
        print("EPOCH:", epoch + 1, "loss:", round(loss_readout, 4))

    # get predictions
    model.eval()
    with torch.no_grad():
        output = model(test_xB_f, test_xB_d, test_xB_t, test_count)
        _, predicted = torch.max(output.data, 1)
    predB = [i.item() for i in predicted]
    predB = removeClassIndicesB(predB)

    # evaluation
    #print("\nTASK B - FINAL RESULTS...")
    x = confusion_matrix(test_yB, predB)
    print("      Predicted:")
    print("          F  T U")
    print("Actual: F", x[0][0], x[0][1], x[0][2])
    print("        T", x[1][0], x[1][1], x[1][2])
    print("        U", x[2][0], x[2][1], x[2][2])
    print("Macro F1:", round(f1_score(test_yB, predB, average="macro"), 4))
    results.append(round(f1_score(test_yB, predB, average="macro"), 4))
    a, b, c = f1_score(test_yB, predB, average=None)
    print("False F1:", round(a, 4))
    print("True F1:", round(b, 4))
    print("Unverified F1:", round(c, 4))
    a = accuracy_score(test_yB, predB)
    print("Accuracy:", round(a, 4))

    all_actual += test_yB
    all_pred += predB

x = confusion_matrix(all_actual, all_pred)
print("=== FINAL ===")
print(x)
print("Final:",round(f1_score(all_actual, all_pred, average="macro"), 4))
a, b, c = f1_score(all_actual, all_pred, average=None)
print("False F1:", round(a, 4))
print("True F1:", round(b, 4))
print("Unverified F1:", round(c, 4))
a = accuracy_score(all_actual, all_pred)
print("Accuracy:", round(a, 4))
