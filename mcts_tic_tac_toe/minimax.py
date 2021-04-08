import math
import numpy as np
from tic_tac_toe import TicTacToeEnv

env = TicTacToeEnv()


def minimax(state, action, turn):
    state_, reward, done, info = env.move(state, action, 1)
    if done:
        return reward

    possible_actions = env.get_action()
    print(possible_actions)

    if turn == 1:
        max_value = -np.inf

    if turn == 0:
        max_value = np.inf

    for i in possible_actions:
        value = minimax(state_, i, abs(1-turn))
        if turn == 1:
            max_value = max(max_value, value)
        else:
            max_value = min(max_value, value)

    return max_value

def search(state):
    board = state
    max_value = -np.inf
    max_action = None
    for i in range(9):
        value = minimax(board, i, 0)
        if value > max_value:
            max_action = i
            max_value = value

    return max_action

state = env.reset()
done = False
while not done:
    action = search(state)
    state_, done, reward, info = env.step(action)
    env.print_board()

    act_o = int(input('choose_action: '))-1
    state_, done, reward, info = env.step(act_o)
    env.print_board()
    state = state_
