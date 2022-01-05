import p5 as p
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
import matplotlib.pyplot as plt
import numpy as np
from tic_tac_toe import TicTacToeEnv

env = TicTacToeEnv()
agent = MCTSVisualiser(env, env.reset(), 1, 1000, 1000)

pix = []
for i in range(1000*1000):
    pix.append(0)
t = 0

def setup():
    p.size(1000,1000)

    font = p.create_font('Montserrat-Medium.ttf', 20)
    p.text_font(font)

    p.background(0)

def draw():
    global pix, t
    p.load_pixels()
    p.pixels = pix.copy()

    agent.reset()
    t = agent.step(t)

#    if p.pixels == pix and t == 50:
#        print('not working')
    p.load_pixels()
    pix = p.pixels.copy()

    t += 1
    print(t)

p.run()
