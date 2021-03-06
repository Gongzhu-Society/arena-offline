import numpy as np
import random

ENCODING_DICT1={'H':0, 'C':1, 'D':2, 'S':3}
ENCODING_DICT2={'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
DECODING_DICT1=['H', 'C', 'D', 'S']
DECODING_DICT2=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
def card_to_vecpos(card):
    #print(card)
    return ENCODING_DICT1[card[0]] * 13 + ENCODING_DICT2[card[1:]]
def vec_to_card(vec):
    #pos=vec.index(1)
    pos = np.where(vec == 1)[0][0]
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]

SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}

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

        self.robot = []


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

    def initial_to_formatted(self, initialcards):
        res=np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res

    def to_state(self, play_order, absolute_history, initial_cards):
        res = absolute_history[play_order].flatten()
        res = np.concatenate((res, absolute_history[(play_order+1)%4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 2) % 4].flatten()))
        res = np.concatenate((res, absolute_history[(play_order + 3) % 4].flatten()))
        res = np.concatenate((res, self.initial_to_formatted(initial_cards)))
        return res

    def judge_winner(self, cards_dans_ce_term):
        gagneur = 0
        for i in [1, 2, 3]:
            if (cards_dans_ce_term[i][0]==cards_dans_ce_term[gagneur][0]) & (ENCODING_DICT2[cards_dans_ce_term[i][1:]]>ENCODING_DICT2[cards_dans_ce_term[gagneur][1:]]):
                gagneur=i
        return gagneur

    def one_turn(self, round, prt):
        # label each player by 0,1,2,3
        cards_in_this_term = []
        input_vec_list=[]
        legal_choice_vec_list = []
        # transform a player's position, absolute history, and initial cards to player's state
        # then let the p-valuation network to decide which one to play

        state_vec1 = self.to_state(self.play_order[0], self.history, self.initial_cards[self.play_order[0]])
        card_played = self.robot[self.play_order[0]].play_one_card([round, cards_in_this_term],state_vec1, self.initial_cards[self.play_order[0]], self.cards_sur_table[self.play_order[0]], 'A', False)
        #print(self.play_order[0], card_played)
        #input_vec_list.append(in_vec)
        cards_in_this_term.append(card_played)
        #legal_choice_vec_list.append(legal_choice_vec)
        couleur_dans_ce_tour = card_played[0]
        #print("card played: ", card_played)
        self.cards_sur_table[self.play_order[0]].append(card_played)
        self.expand_history(card_played, 0, self.play_order[0], round)

        #same thing for the 2nd, 3rd, and 4th player
        for i in range(1, 4):
            #print("player: ", self.play_order[i])
            state_vec1 = self.to_state(self.play_order[i], self.history, self.initial_cards[self.play_order[i]])
            card_played = self.robot[self.play_order[i]].play_one_card([round, cards_in_this_term],state_vec1, self.initial_cards[self.play_order[i]],
                                              self.cards_sur_table[self.play_order[i]], couleur_dans_ce_tour,False)
            #print(self.play_order[i], card_played)
            self.cards_sur_table[self.play_order[i]].append(card_played)
            #print("card played:", card_played)
            self.expand_history(card_played, i, self.play_order[i], round)
            cards_in_this_term.append(card_played)
            #input_vec_list.append(in_vec)
            #legal_choice_vec_list.append(legal_choice_vec)

        # the order of player 0 in this turn
        player_0_order=(4-self.play_order[0])%4
        # add input data into buffer for later training

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
                        if i in SCORE_DICT.keys():
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

        score[0]=score[0]+score[2]
        score[1]=score[1]+score[3]
        score[2]=score[0]
        score[3]=score[1]
        return score

    def one_round(self, prt):
        self.new_shuffle()
        for no_turn in range(13):
            self.one_turn(no_turn, prt)
        #print("Hello here")
        result = self.calc_score()
        return result
    '''

    def train(self, robot, prophet, whole_length, mBuffer, batch_size, epoch, device):
        optimizer1 = optim.Adam(robot.parameters(), lr=0.001, betas=(0.1, 0.999), eps=1e-04, weight_decay=0, amsgrad=False)
        optimizer2 = optim.Adam(prophet.parameters(), lr=0.001, betas=(0.1, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        train_loader = torch.utils.data.DataLoader(dataset=mBuffer, batch_size=batch_size, shuffle=True)

        for jj in range(epoch):


            for i, (input_sample_i, output_sample_i, legal_choices_i, action_i) in enumerate(train_loader):
                #print("i=",i)
                value_prediction = prophet(input_sample_i.reshape(-1, 3016).to(device))
                #print("size of value prediction:", value_prediction.size())
                #print("size of output :", output_sample_i.size())
                output_sample_i = output_sample_i.to(device)
                legal_choices_i = legal_choices_i.to(device)
                action_i = action_i.to(device)

                surprise = output_sample_i-value_prediction[:,0]
                prophet_loss = torch.sum(surprise**2)/batch_size
                loss = robot.loss_func(robot(input_sample_i), surprise, torch.tensor(legal_choices_i), action_i)
                #print(action_i)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                prophet_loss.backward()
                optimizer2.step()
                if i == 0:
                    print(jj / epoch * 100, "pourcent: P loss:", loss, "v loss:", prophet_loss)

    def save_model(self, robot, prophet, robot_path, prophet_path):
        torch.save(robot.state_dict(), robot_path)
        torch.save(prophet.state_dict(), prophet_path)

    def load_model(self, robot, prophet, robot_path, prophet_path):
        robot.load_state_dict(torch.load(robot_path))
        prophet.load_state_dict(torch.load(prophet_path))
        robot.eval()
        prophet.eval()
    '''

from allinone import PNet
from model import Robot
from model import PNet as PNet2
from monsieurrandom import RandomPlayer
from monsieursi import IfPlayer
game = GameRunner()
#robot0 = PNet()
#robot0.load_w("robot-net-0.txt")
#robot2 = PNet()
#robot2.load_w("robot-net-0.txt")
#robot0 = RandomPlayer()
#robot2 = RandomPlayer()
robot0 = IfPlayer()
robot2 = IfPlayer()

ai1 = PNet2()
robot1 = Robot(ai1)
robot1.load_w("robot-net.txt")


ai2 = PNet2()
robot3 = Robot(ai2)
robot3.load_w("robot-net.txt")
game.robot = [robot0, robot1, robot2, robot3]
doc = open('result.txt','w')
print("南北","东西",file=doc)
NOR = 1000
res = np.zeros(NOR)
for i in range(NOR):
    result = game.one_round(False)
    res[i] = result[1]-result[0]
    print(result[0], result[1], file=doc)
    if i%10==3 :
        print("刚刚，颓废的机器人打了", i, "盘")
doc.close()
doc = open("stats.txt",'w')
print("number of rounds",NOR,file=doc)
print("average gain",np.mean(res),file=doc)
print("gain std",np.std(res)/np.sqrt(NOR-1),file=doc)
