import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    #pos=vec.index(1)
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]

SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
TRAINING = True
class PNet(nn.Module):
    def __init__(self):
        #define a model
        super(PNet, self).__init__()
        self.fc1 = nn.Linear(3016, 1482)
        self.fc2 = nn.Linear(1482, 741)
        self.fc3 = nn.Linear(741, 370)
        self.fc4 = nn.Linear(370, 370)
        self.fc5 = nn.Linear(370, 185)
        self.fc6 = nn.Linear(185, 185)
        self.fc7 = nn.Linear(185, 100)
        self.fc8 = nn.Linear(100, 52)

        self.beta = 1

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


    def output_to_probability(self, out_vec, legal_choix):
        expd = torch.exp(self.beta * out_vec)
        expd = torch.mul(expd, torch.tensor(legal_choix))

        prob = expd / torch.sum(expd)
        return prob

    def initial_to_formatted(self, initialcards):
        res=np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def to_my_state(self, play_order, absolute_history, initial_cards):
        res = absolute_history[play_order].flatten()
        res = np.concatenate((res, absolute_history[(play_order+1)%4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 2) % 4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 3) % 4].flatten()))
        res = np.concatenate((res, self.initial_to_formatted(initial_cards)))
        return res

    def to_my_state_li(self,  lichao_history, place, cards_on_table, initial_cards):
        res = np.zeros(2912)
        for i in range(len(lichao_history)):
            round = i//4
            player = lichao_history[i][0]
            player = (player-place+4)%4
            the_card = lichao_history[i][1]
            res[player*728+56*round+4+card_to_vecpos(the_card)] = 1
            res[player*728+56*round+(i%4)] = 1
        rd = len(lichao_history)//4
        for j in range(len(cards_on_table)):
            player = cards_on_table[j][0]
            player = (player - place + 4) % 4
            the_card = cards_on_table[j][1]
            res[player * 728 + 56 * (rd) + 4 + card_to_vecpos(the_card)] = 1
            res[player * 728 + 56 * (rd) + j] = 1
        res = np.concatenate((res, self.initial_to_formatted(initial_cards)))
        return res

    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        #state is already a 1-dim vector
        #returns a np 01 array
        #print("initial vec:", initial_vec)
        #print("played vec:", played_vec)
        whats_left = initial_vec - played_vec
        empty_color = False
        if color_of_this_turn == 'A':
            empty_color = True
        elif whats_left[card_to_vecpos(color_of_this_turn+'2'):card_to_vecpos(color_of_this_turn+'A')].sum() < 1:
            empty_color = True
        if empty_color:
            return whats_left

        pos = np.where(whats_left == 1)[0]
        pos = pos[pos >= card_to_vecpos(color_of_this_turn + '2')]
        pos = pos[pos <= card_to_vecpos(color_of_this_turn + 'A')]

        #print(pos)
        res = np.zeros(52)
        for i in range(len(pos)):
            res[pos[i]] = 1
        return res

    def play_one_card(self, ph, state, initial_cards, cards_played, couleur, training):
        #print("initial cards: ", initial_cards, "cards played:", cards_played)
        legal_choices = self.cards_left(self.initial_to_formatted(initial_cards), self.initial_to_formatted(cards_played), couleur)
        input = np.concatenate((state, legal_choices))
        input = torch.tensor(input)
        n = legal_choices.sum()
        # if there is only one choice, we don't need the calculation of q
        if n < 1.5:
            if training:
                return vec_to_card(legal_choices), input, legal_choices
            else:
                return vec_to_card(legal_choices)
        # elsewise, we need the policy network
        net_output = self.forward(input)

        probability = self.output_to_probability(net_output, legal_choices)
        prb = probability.detach().numpy()
        #print("probability", prb)
        sample_output = np.random.multinomial(1, prb, size=1)
        the_card = vec_to_card(sample_output[0])
        if training:
            return the_card, input, legal_choices
        else:
            return the_card


    def load_w(self, robot_path):
        self.load_state_dict(torch.load(robot_path))
        self.eval()

class VNet(nn.Module):
    def __init__(self):
        #define a model
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(3016, 1482)
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



class GameRunner:
    def __init__(self):
        self.initial_cards = [[]]
        # history from judge's view
        self.history = np.zeros((4, 13, 52+4))
        # another quick save for history
        self.cards_sur_table = [[]]

        self.play_order = [0, 1, 2, 3]
        # 13 elements, representing who wins each turn
        self.who_wins_each_turn = []

    def new_shuffle(self):
        cards = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA',
                 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA',
                 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA',
                 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
        random.shuffle(cards)
        self.play_order = [0, 1, 2, 3]
        random.shuffle(self.play_order)
        for i in [1,2,3]:
            self.play_order[i] = (self.play_order[0]+i)%4
        self.initial_cards = [cards[0:13], cards[13:26],cards[26:39],cards[39:52]]
        self.history = np.zeros((4, 13, 52 + 4))
        self.cards_sur_table=[[],[],[],[]]
        self.who_wins_each_turn = []

    def expand_history(self, card_to_add, pos_in_this_turn, absolute_player, which_round):
        self.history[absolute_player, which_round, 4 + card_to_vecpos(card_to_add)] = 1
        self.history[absolute_player, which_round, pos_in_this_turn] = 1

    def judge_winner(self, cards_dans_ce_term):
        gagneur = 0
        for i in [1, 2, 3]:
            if (cards_dans_ce_term[i][0]==cards_dans_ce_term[gagneur][0]) & (ENCODING_DICT2[cards_dans_ce_term[i][1:]]>ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
                gagneur=i
        return gagneur

    def one_turn(self, round, robot, mBuffer, prt):
        # label each player by 0,1,2,3
        cards_in_this_term = []
        input_vec_list=[]
        legal_choice_vec_list = []
        # transform a player's position, absolute history, and initial cards to player's state
        # then let the p-valuation network to decide which one to play

        state_vec1 = robot.to_my_state(self.play_order[0], self.history, self.initial_cards[self.play_order[0]])
        card_played, in_vec, legal_choice_vec = robot.play_one_card(state_vec1, self.initial_cards[self.play_order[0]], self.cards_sur_table[self.play_order[0]], 'A')
        input_vec_list.append(in_vec)
        cards_in_this_term.append(card_played)
        legal_choice_vec_list.append(legal_choice_vec)
        couleur_dans_ce_tour = card_played[0]
        #print("card played: ", card_played)
        self.cards_sur_table[self.play_order[0]].append(card_played)
        self.expand_history(card_played, 0, self.play_order[0], round)

        #same thing for the 2nd, 3rd, and 4th player
        for i in range(1, 4):
            #print("player: ", self.play_order[i])
            state_vec1 = robot.to_my_state(self.play_order[i], self.history, self.initial_cards[self.play_order[i]])
            card_played, in_vec, legal_choice_vec = robot.play_one_card(state_vec1, self.initial_cards[self.play_order[i]],
                                              self.cards_sur_table[self.play_order[i]], couleur_dans_ce_tour)
            self.cards_sur_table[self.play_order[i]].append(card_played)
            #print("card played:", card_played)
            self.expand_history(card_played, i, self.play_order[i], round)
            cards_in_this_term.append(card_played)
            input_vec_list.append(in_vec)
            legal_choice_vec_list.append(legal_choice_vec)

        # the order of player 0 in this turn
        player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training
        for i in range(4):
            mBuffer.add_input_sample(input_vec_list[(player_0_order+i)%4]) # input vector is pytorch tensor
            mBuffer.add_lc_sample(legal_choice_vec_list[(player_0_order+i)%4]) #legal choices vector list is np array
            mBuffer.add_action_sample(card_to_vecpos(cards_in_this_term[(player_0_order+i)%4])) #action is an integer
        if prt:
            pos_dict = ['南', '东', '北', '西']
            print("第 ", round, " 轮： ", pos_dict[self.play_order[0]], " : ", cards_in_this_term[0], "; ",
                  pos_dict[self.play_order[1]], " : ", cards_in_this_term[1], "; ",
                  pos_dict[self.play_order[2]], " : ", cards_in_this_term[2], "; ",
                  pos_dict[self.play_order[3]], " : ", cards_in_this_term[3], "; ")
            #print("winner:", pos_dict[self.play_order[self.judge_winner(cards_in_this_term)]])
        # judge who wins
        winner = self.play_order[self.judge_winner(cards_in_this_term)]
        self.who_wins_each_turn.append(winner)
        self.play_order = [winner, (winner+1)%4, (winner+2)%4, (winner+3)%4]

    def calc_score(self):
        score = np.zeros(4)
        has_score_flag = [False, False, False, False]
        c10_flag = [False, False, False, False]
        heart_count=np.zeros(4)
        #calc points
        for people in range(4):
            for turn in range(13):
                if self.who_wins_each_turn[turn]==people:
                    for players in range(4):
                        i=vec_to_card(self.history[players, turn])
                        if i=="C10":
                            c10_flag[people]=True
                        else:
                            score[people]+=SCORE_DICT[i]
                            has_score_flag[people]=True
                        if i.startswith('H') or i.startswith('J'):
                            heart_count[people]+=1
            #check whole Hearts
            if heart_count[people]==13:
                score[people]+=400
            # settle transformer
            if c10_flag[people]==True:
                if has_score_flag[people]==False:
                    score[people]+=50
                else:
                    score[people]*=2
        if TRAINING:
            score[0]=score[0]+score[2]
            score[1]=score[1]+score[3]
            score[2]=score[0]
            score[3]=score[1]
        return score

    def one_round(self, robot, mBuffer, prt):
        self.new_shuffle()
        for no_turn in range(13):
            self.one_turn(no_turn, robot, mBuffer, prt)

        # add labels to training data
        result = self.calc_score()
        for i in range(13):
            for j in range(4):
                mBuffer.add_output_sample(result[j])


    def save_model(self, robot, prophet, robot_path, prophet_path):
        torch.save(robot.state_dict(), robot_path)
        torch.save(prophet.state_dict(), prophet_path)

    def load_model(self, robot, prophet, robot_path, prophet_path):
        robot.load_state_dict(torch.load(robot_path))
        prophet.load_state_dict(torch.load(prophet_path))
        robot.eval()
        prophet.eval()



