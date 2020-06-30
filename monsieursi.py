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
def vecpos_to_card(pos):
    return DECODING_DICT1[pos//13]+DECODING_DICT2[pos%13]
SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}

class IfPlayer:
    def cards_left(self, initial_vec, played_vec, color_of_this_turn):
        whats_left = initial_vec - played_vec
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

        #print(pos)
        res = np.zeros(52)
        for i in range(len(pos)):
            res[pos[i]] = 1
        return res

    def initial_to_formatted(self, initialcards):
        res=np.zeros(52)
        for i in initialcards:
            res[card_to_vecpos(i)] = 1
        return res
    def play_one_card(self, info, state, initial_cards, cards_played, couleur, training):
        legal_choices = self.cards_left(self.initial_to_formatted(initial_cards),
                                        self.initial_to_formatted(cards_played), couleur)
        which_turn = info[0]
        cards_in_this_turn = info[1]
        if len(cards_in_this_turn)==2:
            #别人出猪圈我出猪
            if cards_in_this_turn[1] == 'SA' or (cards_in_this_turn[1] == 'SK' and cards_in_this_turn[0] != 'SA'):
                if legal_choices[card_to_vecpos('SQ')]==1:
                    return 'SQ'
            #我断门了，而且对家小，垫大牌
            if legal_choices[card_to_vecpos(couleur + '2'):(card_to_vecpos(couleur + 'A') + 1)].sum() < 1:
                if (cards_in_this_turn[1][0] == cards_in_this_turn[0][0] and ENCODING_DICT2[cards_in_this_turn[1][1:]] > ENCODING_DICT2[cards_in_this_turn[0][1:]]) :
                    if legal_choices[card_to_vecpos('SQ')] == 1:
                        return 'SQ'
                    elif legal_choices[0:13].sum() >0.5:
                        pos = np.where(legal_choices[0:13] == 1)[0]
                        return vecpos_to_card(pos[-1])

        elif len(cards_in_this_turn)==3:
            #别人出猪圈我出猪
            if cards_in_this_turn[0] == 'SA' or cards_in_this_turn[2] == 'SA' or \
                    (cards_in_this_turn[0] == 'SK' and cards_in_this_turn[1] != 'SA') or \
                    (cards_in_this_turn[2] == 'SK' and cards_in_this_turn[1] != 'SA'):
                if legal_choices[card_to_vecpos('SQ')]==1:
                    return 'SQ'
            #我断门了，而且对家小，垫大牌
            if legal_choices[card_to_vecpos(couleur + '2'):(card_to_vecpos(couleur + 'A') + 1)].sum() < 1:
                if (cards_in_this_turn[1][0] == cards_in_this_turn[0][0] and
                        (ENCODING_DICT2[cards_in_this_turn[0][1:]] > ENCODING_DICT2[cards_in_this_turn[1][1:]] or
                         ENCODING_DICT2[cards_in_this_turn[2][1:]] > ENCODING_DICT2[cards_in_this_turn[1][1:]])) or\
                        (cards_in_this_turn[1][0] != cards_in_this_turn[0][0]):
                    if legal_choices[card_to_vecpos('SQ')] == 1:
                        return 'SQ'
                    elif legal_choices[0:13].sum() >0.5:
                        pos = np.where(legal_choices[0:13] == 1)[0]
                        return vecpos_to_card(pos[-1])





        # if your teammate is not going to win, give the greatest point card
        # if your teammate is sure to win, give, give the greatest point card
        prb = legal_choices/np.sum(legal_choices)
        sample_output = np.random.multinomial(1, prb, size=1)
        the_card = vec_to_card(sample_output[0])
        return the_card