import math
import numpy as np
from tic_tac_toe import TicTacToeEnv

class Node:
    def __init__(self, parent, possible_actions, turn, player):
        self.state = None
        self.parent = parent
        self.player = player
        self.value = 0
        self.visits = 0
        self.children = [None for i in range(9)]
        self.possible_actions = possible_actions
        self.fully_expanded = False
        self.updated_node = True
        self.turn = turn
        self.terminal = False

    @property
    def depth(self):
        count = 0
        node = self
        while not node.parent is None:
            count += 1
            node = node.parent

        return count

    def expand(self, env, turn):
        if self.fully_expanded or not self.updated_node:
            print('the node is already fully expanded or its not updated')

        state_1 = self.state.copy()
        playa = self.player

        for action in self.possible_actions:
            self.state = state_1
            next_state, reward, done = env.move(self.state, action, self.player, turn)
            next_possible_actions = self.possible_actions.copy()
            next_possible_actions.remove(action)

            node = Node(self, next_possible_actions, self.turn, abs(1-playa))
            if done or len(next_possible_actions) == 0:
                node.terminal = True
            node.state = next_state
            node.updated_node = True
            self.children[action] = node

        self.player = playa
        flag = False
        for i in self.possible_actions:
            if self.children[i] is None:
                flag = True
        if not flag:
            self.fully_expanded = True

    def choose_node(self, scalar):
        max_ucb = -np.inf
        max_node = []
        ucbs = []

        for action in self.possible_actions:
            child = self.children[action]
            if child.visits > 0:
                ucb = child.value / child.visits + scalar * math.sqrt(math.log(self.visits / child.visits))
            else:
                ucb = np.inf
            ucbs.append(ucb)
            if ucb > max_ucb:
                max_ucb = ucb
                max_node.append(child)

        return np.random.choice(max_node)

    def choose_action(self, scalar):
        max_ucb = -np.inf
        max_action = []

        for action in self.possible_actions:
            child = self.children[action]
            if child.visits > 0:
                ucb = child.value / child.visits + scalar * math.sqrt(math.log(self.visits / child.visits))
            else:
                ucb = np.inf
            if ucb > max_ucb:
                max_ucb = ucb
                max_action.append(action)

        return np.random.choice(max_action)

    def backpropogate(self, value):
        node = self
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent


class MCTS:
    def __init__(self, env):
        self.env = env

    def search(self, init_state, turn):
        possible_actions = self.env.possible_actions()
        start_node = Node(None, possible_actions, turn, turn)
        start_node.state = init_state

        for alo in range(1000):
            node = start_node

            # traverse
            if node.fully_expanded:
                while node.fully_expanded:
                    node = node.choose_node(1/math.sqrt(2))


            node_terminaled = False
            if node.terminal:
                #print(node.state)
                #print('terminaled')
                action = node.parent.children.index(node)
                _, reward, _  = self.env.move(node.parent.state, action, node.parent.player, turn)
                node.backpropogate(reward)
                node_terminaled = True
                continue

            # expand
            node.expand(self.env, turn)
            node = node.choose_node(2)

            if node.terminal and not node_terminaled:
                #print('terminaled')
                action = node.parent.children.index(node)
                _, reward, _ = self.env.move(node.parent.state, action, node.parent.player, turn)
                node.backpropogate(reward)
                continue

            # rollout
            player = node.player
            done = False
            possible_actions = node.possible_actions.copy()
            state = node.state
            while not done:
                if len(possible_actions) > 0:
                    #print(player, 'player')
                    action = np.random.choice(possible_actions)
                    state, reward, done = self.env.move(
                        state,
                        action,
                        player,
                        turn
                    )
                    #env.print_board(state)
                    #if done:
                        #print(reward)
                    possible_actions.remove(action)

                else:
                    done = True
                    reward = 0

                if done:
                    #print('backpropp')
                    #env.print_board(state)
                    node.backpropogate(reward)

                player = abs(1-player)


        print(start_node.value, start_node.visits)
        return start_node.choose_action(1/math.sqrt(2))
