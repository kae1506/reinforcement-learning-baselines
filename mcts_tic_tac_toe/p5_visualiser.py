import p5 as p
from mcts import MCTSVisualiser
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