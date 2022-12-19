import random
import chess
import chess.engine
import numpy as np
import matplotlib.pyplot as plt
import pickle
import reward_chess

class RandomAgent():
    def get_action(self, board):
        moves = [x for x in board.generate_legal_moves()]
        return np.random.choice(moves)

class MinMaxAgent():
    def alphabeta(self, board, alpha, beta, depthleft):
        bestscore = -9999
        if( depthleft == 0 ):
            return self.quiesce(board, alpha, beta)
        for move in board.legal_moves:
            board.push(move)   
            score = -self.alphabeta(board, -beta, -alpha, depthleft - 1)
            board.pop()
            if( score >= beta ):
                return score
            if( score > bestscore ):
                bestscore = score
            if( score > alpha ):
                alpha = score   
        return bestscore

    def quiesce(self, board, alpha, beta ):
        stand_pat = reward_chess.evaluate_board(board)
        if( stand_pat >= beta ):
            return beta
        if( alpha < stand_pat ):
            alpha = stand_pat

        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)        
                score = -self.quiesce(board, -beta, -alpha )
                board.pop()

                if( score >= beta ):
                    return beta
                if( score > alpha ):
                    alpha = score  
        return alpha

    def get_action(self, board, depth):
        bestMove = chess.Move.null()
        bestValue = -99999
        alpha = -100000
        beta = 100000
        for move in board.legal_moves:
            board.push(move)
            boardValue = -self.alphabeta(board, -beta, -alpha, depth-1)
            if boardValue > bestValue:
                bestValue = boardValue;
                bestMove = move
            if( boardValue > alpha ):
                alpha = boardValue
            board.pop()
        return bestMove

# Constants for epsilon-greedy policy
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# Constants for Q-learning
ALPHA = 0.1
GAMMA = 0.9

class QLearningAgent:
    def __init__(self):
        self.q_values = {}
        self.epsilon = EPSILON_START

    def get_legal_actions(self, board):
        # Returns a list of all legal moves that can be made from the current board position
        legal_actions = []
        for move in board.legal_moves:
            legal_actions.append(move)
        return legal_actions

    def get_max_q_value(self, board):
        # Returns the maximum Q-value for all legal actions in the current board position
        max_q_value = -float('inf')
        legal_actions = self.get_legal_actions(board)
        for action in legal_actions:
            q_value = self.get_q_value(board, action)
            if q_value > max_q_value:
                max_q_value = q_value
        return max_q_value

    def get_q_value(self, board, action):
        # Returns the Q-value for the given board and action
        if (board.fen(), action) not in self.q_values:
            self.q_values[(board.fen(), action)] = 0.0
        return self.q_values[(board.fen(), action)]

    def get_action(self, board):
        if board.is_game_over():
            return
        if random.random() < self.epsilon:
            # Choose a random action
            legal_actions = self.get_legal_actions(board)
            action = random.choice(legal_actions)
        else:
            max_q_value = -float('inf')
            best_actions = []
            legal_actions = self.get_legal_actions(board)
            for action in legal_actions:
                q_value = self.get_q_value(board, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_actions = [action]
                elif q_value == max_q_value:
                    best_actions.append(action)
            action = random.choice(best_actions)
        return action
    def update(self, board, action, reward, next_board):
        # Updates the Q-value
        next_q_value = self.get_max_q_value(next_board)
        q_value = self.get_q_value(board, action)
        self.q_values[(board.fen(), action)] = q_value + ALPHA * (reward + GAMMA * next_q_value - q_value)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


class RepeatedUpdateQLearningAgent(QLearningAgent):
    def __init__(self):
        super().__init__()
        self.num_repeats = 0
    
    def update(self, board, action, reward, next_board):
        # Repeated update Q-learning algorithm
        self.num_repeats = int(1.0 / self.epsilon)
        for _ in range(self.num_repeats):
            next_q_value = self.get_max_q_value(next_board)
            q_value = self.get_q_value(board, action)
            self.q_values[(board.fen(), action)] = q_value + ALPHA * (reward + GAMMA * next_q_value - q_value)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)








