import math
import numpy as np
import p5 as p
from tic_tac_toe import TicTacToeEnv
from copy import deepcopy

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
        print('expand')
        if self.fully_expanded or not self.updated_node:
            print(self.fully_expanded)
            print(self.updated_node)
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

    def choose_node(self, scalar, turn):
        if self.player != turn:
            return self.choose_node_positive(scalar)
        else:
            return self.choose_node_negative(scalar)

    def choose_node_positive(self, scalar):
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

    def choose_node_negative(self, scalar):
        max_ucb = np.inf
        max_node = []
        ucbs = []

        for action in self.possible_actions:
            child = self.children[action]
            if child.visits > 0:
                ucb = -(child.value / child.visits + scalar * math.sqrt(math.log(self.visits / child.visits)))
            else:
                ucb = -np.inf
            ucbs.append(ucb)
            if ucb < max_ucb:
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
            node = node.choose_node(2, turn)

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


class MCTSVisualiser:
    def __init__(self, env, init_state, turn, width, height):
        self.env = env
        self.possible_actions = self.env.possible_actions()
        self.start_node = Node(None,self.possible_actions, turn, turn)
        self.start_node.state = init_state
        self.on_screen_depths = 0
        self.scroll = 0
        self.node = self.start_node
        self.turn = turn
        self.width = width
        self.height = height

        self._xy = (0,0)

    def position(self,node):
        y = 100 + 100*node.depth
        offset = 25
        if node.parent:
            index = node.parent.children.index(node)
            x = (int((self.width-(50*len(self.possible_actions) + offset*len(self.possible_actions)))/2)+50/2) + ((25*3) * index)
        else:
            x = self.width/2
        return x, y


    def reset(self):
        p.stroke_weight(3)
        p.fill(255,0,0)
        p.stroke(255)
        p.rect_mode(p.CENTER)

    def render(self, node):
        self.reset()
        p.ellipse(self.width/2, 100, 50, 50)
        p.text_align(p.CENTER)

        p.stroke(255)
        p.stroke_weight(1)
        p.fill(255)
        p.text(f'{node.visits} {node.value}', self.width/2-50/2+20, 100 + 50)
        self.reset()

        self._xy = (self.width/2, 100)

    def render_children(self, children, chosen_node, parent, ids=None):
        offset = 25
        last_pos = (int((self.width-(50*len(self.possible_actions) + offset*len(self.possible_actions)))/2)+50/2)
        chosen_xy = (0,0)

        if ids:
            print(ids)

        for i in children:
            if i == chosen_node:
                p.stroke(0,255,0)
                p.fill(0,255,0)

            if i == None:
                p.fill(0,0,0,50)

                p.line(last_pos, 100 + 100*chosen_node.depth + self.scroll, self._xy[0], self._xy[1])
                p.ellipse(last_pos, 100 + 100*chosen_node.depth + self.scroll, 50, 50)

            else:
                p.line(last_pos, 100 + 100*i.depth + self.scroll, self._xy[0], self._xy[1])
                p.ellipse(last_pos, 100 + 100*i.depth + self.scroll, 50, 50)

            if i is not None:
                p.stroke(255)
                p.fill(255)
                p.stroke_weight(1)

                p.text(f'{i.visits} {i.value}', last_pos, 100+100*i.depth+self.scroll+25)

            self.reset()

            if i == chosen_node:
                chosen_xy = (last_pos, 100 + 100*i.depth + self.scroll)

            last_pos += offset + offset + 50/2

        self._xy = deepcopy(chosen_xy)
        print(self._xy)

    def render_rollout(self, node, reward):
        p.stroke(0,0,255)
        p.stroke_weight(5)
        print(self._xy)

        p.line(self._xy[0], self._xy[1], self._xy[0], self._xy[1]+100)
        p.line(self._xy[0], self._xy[1]+100, self._xy[0]-15, self._xy[1]+100-15)
        p.line(self._xy[0], self._xy[1]+100, self._xy[0]+15, self._xy[1]+100-15)

        p.stroke(255)
        p.fill(255)
        p.stroke_weight(1)
        p.text(f"R:{reward}", self.width/2, 900-self.scroll*5+25)

        self.reset()

    def render_backpropogation(self, node, reward):
        node_ = node
        while node_ is not None:
            x, y = self.position(node_)
            p.stroke(0,255,0)
            p.fill(255,0,0)
            p.ellipse(x, y, 50, 50)
            if node_.parent:
                px, py = self.position(node_.parent)
                p.line(px, py, x, y)

            node_ = node_.parent
            self.reset()


    def step(self, t):
        node = self.node

        if t == 0 or t == 1:
            p.fill(0)
            p.stroke(0)
            node = self.start_node
            p.rect(500,500, self.width, self.height)

            self.reset()
            self.render(node)


        self.reset()
    #    node = self.start_node

        # traverse
        if t == 25:
            if node.fully_expanded:
                while node.fully_expanded:
                    new_node = node.choose_node(1/math.sqrt(2), self.turn)
                    self.render_children(node.children, new_node, node, ids='traverse__')
                    node = new_node
                    self.reset()

            p.fill(255)
            p.stroke(255)
            p.stroke_weight(1)
            p.text('traverse', 900,800)

            self.reset()

        node_terminaled = False
        if node.terminal:
            #print(node.state)
            print('terminaled')
            action = node.parent.children.index(node)
            _, reward, _  = self.env.move(node.parent.state, action, node.parent.player, turn)
            #self.render_backpropogation(node, reward)
            node.backpropogate(reward)
            node_terminaled = True
            t = 0
            return

        # expand
        if t == 50:
            node.expand(self.env, self.turn)
            new_node = node.choose_node(2, self.turn)
            self.render_children(node.children, new_node, node, ids='expand__')
            node = new_node
            p.fill(255)
            p.stroke(255)
            p.stroke_weight(1)
            p.text('expansion', 900,850)

        if node.terminal and not node_terminaled:
            #print('terminaled')
            action = node.parent.children.index(node)
            _, reward, _ = self.env.move(node.parent.state, action, node.parent.player, turn)
            #self.render_backpropogation(node, reward)
            node.backpropogate(reward)
            t = 0
            return t

        # rollout
        if t == 75:
            player = node.player
            done = False
            possible_actions = node.possible_actions.copy()
            state = node.state
            p.fill(255)
            p.stroke(255)
            p.stroke_weight(1)
            p.text('rollout', 900,900)

            self.reset()

            while not done:
                if len(possible_actions) > 0:
                    #print(player, 'player')
                    action = np.random.choice(possible_actions)
                    state, reward, done = self.env.move(
                        state,
                        action,
                        player,
                        self.turn
                    )
                    #env.print_board(state)
                    #if done:
                        #print(reward)
                    possible_actions.remove(action)

                else:
                    done = True
                    reward = 0

                if done:
                    self.render_rollout(node, reward)
                    #self.render_backpropogation(node, reward)
                    node.backpropogate(reward)


                player = abs(1-player)

        if t == 125:
            t = 0

        self.node = node

        return t