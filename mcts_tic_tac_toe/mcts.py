import numpy as np
import math
import random
import heartrate

class Node:
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.player = None
        self.children = {}

        self.value = 0
        self.visits = 0

        self.is_expanded = False
        if self.state.is_terminal() == True:
            self.is_terminal = True
        
    def choose_node(self, exploration_constant):
        best_ucb = float('-inf')
        best_node = None

        for child in self.children.values():
            if child.visits > 0:
                ucb = child.value / child.visits + exploration_constant* \
                        math.sqrt(math.log(self.visits/child.visits))
            else:
                ucb = float('inf')

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child

        return best_node

    def expand(self):
        pass

from copy import deepcopy

class Board:
    def __init__(self, board=None):
        if board:
            self.state = deepcopy(board.state)
            self.p1 = deepcopy(board.p1)

        else:
            self.state = ['#'] * 9
            self.p1 = 1


    def generate_states(self, state=None):
        if state is None:
            state = self

        states = []
        # print(state.p1)

        for i in range(9):
            if state.state[i] == '#':
                board = Board(state)
                board.p1 = 3-state.p1
                # print(board.p1, 'p1')
                states.append(board.make_move(i))

        # print(states[1].p1)
        # quit()
        return states

    def make_move(self, position):
        board = Board(self)

        board.state[position] = board.p1
        
        return board

    def is_tie(self):
        for i in self.state:
            if i == '#':
                return False

        return True

    def check(self, state=None):
        if state is None:
            state = self.state

        for i in range(3):
            if state[i*3] == state[i*3 + 1] == state[i*3 + 2] and state[i*3] != '#':
                return state[i*3]

        for i in range(3):
            if state[i] == state[i+3] == state[i+6] and state[i] != '#':
                return state[i]

        if state[0] == state[4] == state[8] and state[0] != '#':
            return state[0]

        if state[2] == state[4] == state[6] and state[2] != '#':
            return state[2]

        if self.is_tie():
            return 0

        return 'False'

    def get_winner(self):
        if self.check() not in [0, 1, 2]:
            print('called get_winner when it wasnt terminal, use is_terminal instead')

        return self.check()

    def is_terminal(self):
        if self.check() in [0, 1, 2]:
            return True

        return False
            
class MCTS:
    def __init__(self, iterations=1600):
        self.iterations = iterations
        self.tree = None

    def search(self, starting_board, player):

        opponent = 3-player
        self.tree = Node(None, starting_board)
        self.tree.player = opponent

        for iteration in range(self.iterations):
            node = self.traverse_and_expand(self.tree)
            
            score = self.rollout(node, opponent)
            self.backpropogate(node, score)

        winner_node = self.tree.choose_node(0)
        # print(self.tree.state.is_terminal())
        # for i in list(self.tree.children.keys()) : 
        #     print(len(self.tree.children.keys()))
        #     print(i.state)        
        return winner_node.state

    def traverse_and_expand(self, node):
        while not node.state.is_terminal():
            if node.is_expanded:
                node = node.choose_node(2)
                # print(node.choose_node(2))
            else:
                return self.expand(node)

        return node

    def expand(self, node):
        states = node.state.generate_states()

        for state in states:
            child = Node(node, state)

            node.children[state] = child
            node.children[state].player = 3 - node.player

        node.is_expanded = True
        return random.choice(list(node.children.values()))

    def rollout(self, node, opponent):
        temp_node = Node(node.parent, node.state)
        temp_node.player = node.player
        if temp_node.state.is_terminal():
            status = temp_node.state.get_winner()

            if status == opponent:
                temp_node.parent.value = -10000
                return status

        board = temp_node.state
        while not board.is_terminal():
            board = random.choice(board.generate_states())
        

        return board.get_winner()
        
    def backpropogate(self, node, result):
        temp_node = node
        while (temp_node != None):
            temp_node.visits += 1
            if (temp_node.player == result):
                temp_node.value += 10
            temp_node = temp_node.parent

if __name__ == '__main__':
    mcts = MCTS()
    b = Board()
    b.p1 = 2
    if int(input('1 if heartrate trace')) == 1:
        heartrate.trace(browser=True)

    p = 1
    while not b.is_terminal():
        print(b.p1, ' stot')

        b = mcts.search(b, p)




        print('\n\n')
        for i in range(3):
            for j in range(3):
                print(b.state[i*3 + j], end=' ')
            print('\n')

        p = 3-p
        # b = mcts.search(b, p)
        # p = 3-p
        # print('\n\n')
        # for i in range(3):
        #     for j in range(3):
        #         print(b.state[i*3 + j], end=' ')
        #     print('\n')
        