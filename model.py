import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import pygame
import random

ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
def card_to_vecpos(card):
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]

def vec_to_card(vec):
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]
def vec_to_cards(vec):
    pos = np.where(vec == 1)[0]
    res = []
    for i in pos:
        res.append(DECODING_DICT1[i//13]+DECODING_DICT2[i%13])
    return res

class PNet(nn.Module):

    def __init__(self):
        #define a model
        super(PNet, self).__init__()
        self.fc1 = nn.Linear(2964, 1482)
        self.fc2 = nn.Linear(1482, 741)
        self.fc3 = nn.Linear(741, 370)
        self.fc4 = nn.Linear(370, 370)

        self.fc41 = nn.Linear(370, 370)
        self.fc42 = nn.Linear(370, 370)
        self.fc43 = nn.Linear(370, 370)
        self.fc44 = nn.Linear(370, 370)

        self.fc5 = nn.Linear(370, 185)
        self.fc6 = nn.Linear(185, 185)
        self.fc7 = nn.Linear(185, 100)
        self.fc8 = nn.Linear(100, 52)



    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x1 = F.relu(self.fc4(x))
        x = F.relu(self.fc41(x))
        x = F.relu(self.fc42(x))
        x = F.relu(self.fc43(x))
        x = F.relu(self.fc44(x))

        x = F.relu(self.fc5(x+x1))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

class Robot():
    def __init__(self, pnet):
        self.pnet = pnet
        self.beta = 1

    def loss_func_single(self, features, output, legal_choix):
        #print("hello here")
        alpha = 0.001
        bc = 0.0000001
        expd = torch.exp(self.beta * features)
        expd = torch.mul(expd, legal_choix)

        prob = expd / torch.sum(expd)
        similarity = -torch.sum(torch.mul(torch.log(prob+0.000001), output))
        entropy = torch.sum(torch.mul(prob, torch.log(prob+0.0000001)))


        return alpha * similarity + bc * entropy

    def loss_func(self, features_v, output_v, legal_choix_v):
        res = torch.zeros(len(output_v))
        for i in range(0, len(output_v)):
            features = features_v[i]
            output = output_v[i]
            legal_choix = legal_choix_v[i]
            #print("bonjour")
            res[i] = self.loss_func_single(features, output, legal_choix)
            #print("bonjour2")
        return torch.sum(res)/(len(output_v))

    def output_to_probability(self, out_vec, legal_choix):
        expd = torch.exp(self.beta * out_vec)
        #legal_choix_r.to(device)
        expd = torch.mul(expd, legal_choix)
        prob = expd / torch.sum(expd)
        return prob

    def initial_to_formatted(self, initialcards):
        res = np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        whats_left = initial_vec - played_vec
        #print("whata left are", vec_to_cards(whats_left))
        empty_color = False
        if color_of_this_turn == 'A':
            empty_color = True

        elif whats_left[card_to_vecpos(color_of_this_turn+'2'):(card_to_vecpos(color_of_this_turn+'A')+1)].sum() < 1:
            empty_color = True
        if empty_color:
            return whats_left

        pos = np.where(whats_left == 1)[0]
        pos = pos[pos >= card_to_vecpos(color_of_this_turn + '2')]
        pos = pos[pos <= card_to_vecpos(color_of_this_turn + 'A')]

        res = np.zeros(52)
        for i in range(len(pos)):
            res[pos[i]] = 1
        return res

    def play_one_card(self,ph, state, initial_cards, cards_played, couleur, training):
        legal_choices = self.cards_left(self.initial_to_formatted(initial_cards), self.initial_to_formatted(cards_played), couleur)

        input = torch.tensor(state)
        #input = torch.tensor(input)
        n = legal_choices.sum()
        # if there is only one choice, we don't need the calculation of q
        if n < 1.5:
            sample_output = np.random.multinomial(1, legal_choices, size=1)
            the_card = vec_to_card(sample_output[0])
            return the_card
            #return legal_choices
        # elsewise, we need the policy network
        net_output = self.pnet(input)

        probability = self.output_to_probability(net_output, torch.tensor(legal_choices))
        device = 'cpu'
        if device == 'cpu':
            prb = probability.detach().numpy()
        else:
            prb = probability.detach().cpu().numpy()

        #sample_output = np.random.multinomial(1, prb, size=1)
        sample_output = np.random.multinomial(1, prb, size=1)
        the_card = vec_to_card(sample_output[0])
        return the_card
    def load_w(self, robot_path):
        self.pnet.load_state_dict(torch.load(robot_path))
        self.pnet.eval()

class Memory(torch.utils.data.Dataset):
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.input_samples = []
        self.target_policy = []
        self.target_value = []
        self.legal_choices = []

        self.first_empty_i = 0
        self.first_empty_tp = 0
        self.first_empty_tv = 0
        self.first_empty_lc = 0

    def __getitem__(self, index):
        return self.input_samples[index], self.target_policy[index], self.target_value[index], self.legal_choices[index]

    def __len__(self):
        return self._max_memory

    def add_input_sample(self, input_sample):
        self.input_samples.append(input_sample)
        self.first_empty_i += 1
        if self.first_empty_i == self._max_memory:
            self.first_empty_i -= 1

    def add_target_policy(self, tgt_policy):
        self.target_policy.append(tgt_policy)
        self.first_empty_tp += 1
        if self.first_empty_tp == self._max_memory:
            self.first_empty_tp -= 1

    def add_target_value(self, tgt_value):
        self.target_value.append(tgt_value)
        self.first_empty_tv += 1
        if self.first_empty_tv == self._max_memory:
            self.first_empty_tv -= 1

    def add_lc_sample(self, lc):
        self.legal_choices.append(lc)
        self.first_empty_lc +=1
        if self.first_empty_lc == self._max_memory:
            self.first_empty_lc -=1

    def clear(self):
        self.first_empty_i = 0
        self.first_empty_tv = 0
        self.first_empty_tp = 0
        self.first_empty_lc = 0
        length = len(self.target_value)
        for i in range(length):
            self.input_samples.pop()
            self.target_policy.pop()
            self.target_value.pop()
            self.legal_choices.pop()

class VNet(nn.Module):
    def __init__(self):
        #define a model
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(2964, 1482)
        self.fc2 = nn.Linear(1482, 741)
        self.fc3 = nn.Linear(741, 370)
        self.fc4 = nn.Linear(370, 185)
        self.fc5 = nn.Linear(185, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 20)
        self.fc8 = nn.Linear(20, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x





class Prophet():
    def __init__(self, vnet):
        self.vnet = vnet

    def loss_func(self, output_v, target_v):
        res = torch.zeros(len(output_v))
        for i in range(0, len(output_v)):
            res[i] = (output_v[i]-target_v[i])*(output_v[i]-target_v[i])
        return torch.sum(res)/(len(target_v))