from p5 import *
import random
from tic_tac_toe import TicTacToeEnv
from mcts import MCTS

def setup():
    size(1000, 1000)

time = 0
def draw():
    global time
    color_offset = time*10
    background(0,0,0)
    translate(0, 0)

    rect_mode(CENTER)
    fill(255, 0, 0)
    stroke(255)

    offset = 25
    last_pos = (int((1000-(50*10 + offset*10))/2)+50/2)
    action = 9
    for i in range(10):
        if i == action:
            stroke(0,255,0)
            fill(0,255,0)
        line(500, 100, last_pos, 300)
        ellipse(last_pos, 300, 50, 50)
        if i == action:
            stroke(255)
            fill(255,0,0)
        last_pos += offset + offset + 50/2

    ellipse(500, 100, 50, 50)
    stroke_weight(3)

    offset = 25
    pos = (int((1000-(50*9 + offset*9))/2)+50/2)
    action = 0
    for i in range(9):
        if i == action:
            stroke(0,255,0)
            fill(0,255,0)
        line(last_pos - offset - 50/2, 300, pos, 600)
        ellipse(pos, 600, 50, 50)
        if i == action:
            fill(255,0,0)
            stroke(255)
        pos += offset + offset + 50/2

    offset = 25
    a_pos = (int((1000-(50*8 + offset*8))/2)+50/2)
    action = 4
    for i in range(8):
        if i == action:
            stroke(0,255,0)
            fill(0,255,0)
        line((int((1000-(50*9 + offset*9))/2)+50/2), 600, a_pos, 900)
        ellipse(a_pos, 900, 50, 50)
        if i == action:
            fill(255,0,0)
            stroke(255)
        a_pos += offset + offset + 50/2


run()
