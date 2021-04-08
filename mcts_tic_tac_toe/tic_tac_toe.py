import math
import random

class TicTacToeEnv:
    def __init__(self):
        self.turn = 1
        self.board = ['#'] * 9
        self.is_over = False

    def print_board(self, state=None):
        if state == None:
            state = self.board

        board = ''
        for i in range(3):
            row = ''
            for j in range(3):
                row += f' {state[i*3+j]}'
            board += f'{row} \n'

        print(board)

    def reset(self):
        self.turn = 1
        self.board = ['#'] * 9
        self.is_over = False

        return self.board

    def step(self, action):
        ''' classic do move '''
        print('alo')
        reward = 0
        done = False
        if self.board[action] == '#':
            self.board[action] = self.turn

        winner = self.get_winner()
        print(winner)

        if winner is not None:
            done = True
            print(f'{winner} won!')
            if winner == self.turn:
                reward = 1
            else:
                reward = -1

        if self.is_tie(self.board):
            done = True
            reward = 0


        self.turn = abs(1-self.turn)


        return self.board, reward, done, {}

    def get_winner(self):
        ''' returns winner, if any '''


        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] and self.board[i] != '#':
                self.is_over = True
                return self.board[i]

        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] and self.board[i] != '#':
                self.is_over = True
                return self.board[i]

        if self.board[0] == self.board[4] == self.board[8] and self.board[0] != '#':
            self.is_over = True
            return self.board[0]

        if self.board[2] == self.board[4] == self.board[6] and self.board[2] != '#':
            self.is_over = True
            return self.board[2]

        return None

    def is_tie(self, board):
        flag = True
        for pos in board:
            if pos == '#':
                flag = False
        if flag:
            print('flagging')
        return flag

    def possible_actions(self):
        possible_actions = []
        for i in range(len(self.board)):
            if self.board[i] == '#':
                possible_actions.append(i)

        return possible_actions

    def move(self, state, action, player, turn):
        ''' basically step function, but doesnt affect the global board'''
        done = False
        reward = 0

        board = self.board.copy()
        self.board = state.copy()
        if self.board[action] == '#':
            self.board[action] = player

        ret_board = self.board.copy()
        winner = self.get_winner()
        self.board = board

        if winner is not None:
            done = True
            if winner == turn:
                reward = 1
            else:
                reward = -1

        if self.is_tie(self.board):
            done = True
            reward = 0

        return ret_board, reward, done

if __name__ == '__main__':
    env = TicTacToeEnv()
    env.print_board()
    done = False
    while not done:
        action = int(input(f'{env.turn} turn! enter action (1 to 9): ')) - 1
        _, done, _, _ = env.step(action)
        env.print_board()
