import chess
import agents
import random
import pickle
import reward_chess
import numpy as np

def train_agent(agent, episodes=10):
    board = chess.Board()

    average_q_values = []
    min_q_values = []
    max_q_values = []
    list_cumulative_reaward = []
    wins = 0
    losses = 0
    draws = 0

    engine = chess.engine.SimpleEngine.popen_uci("stockfish_15_x64_avx2.exe")
    for episode in range(episodes):
        
        board = chess.Board()
        reward = 0
        cumulative_reward = 0

        # Choose the color of the agent randomly
        if random.random() < 0.5:
            agent_color = chess.WHITE
            engine_color = chess.BLACK
        else:
            agent_color = chess.BLACK
            engine_color = chess.WHITE

        while not board.is_game_over():
            if board.turn == agent_color:
                # Get the action
                action = agent.get_action(board)

                # Make the move and get the reward
                board.push(action)
                reward = reward_chess.evaluate_board(board)
                cumulative_reward += reward
                next_board = board.copy()

                # Update the Q-value
                agent.update(board, action, reward, next_board)

                # Set the board to the next board state
                board = next_board
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

        # Update the win/loss/draw counts
        result = board.result()

        if agent_color == chess.BLACK:
            if result == "1-0":
                losses += 1
            elif result == "0-1":
                wins += 1
            elif result == "1/2-1/2":
                draws += 1
        else : 
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            elif result == "1/2-1/2":
                draws += 1

        total_q_value = 0
        temp_q_values = []
        for key in agent.q_values:
            total_q_value += agent.q_values[key]
            temp_q_values.append(agent.q_values[key])
        min_q_value = np.min(temp_q_values)
        max_q_value = np.max(temp_q_values)
        temp_q_value = total_q_value / len(agent.q_values)

        # Append the average Q-value to the list
        average_q_values.append(temp_q_value)
        max_q_values.append(max_q_value)
        min_q_values.append(min_q_value)

        list_cumulative_reaward.append(cumulative_reward)

        print("Episode =>", episode ,"Total reward:", cumulative_reward, "Epsilon:", agent.epsilon, "Wins:", wins, "Losses:", losses, "Draws:", draws)

    if type(agent) is agents.QLearningAgent:
        # Save the Q-learning agent to a file
        with open("q_learning_agent.pkl", "wb") as f:
            pickle.dump(agent, f)
    else:
        # Save the Repeated UpdateQ-learning agent to a file
        with open("repeated_update_q_learning_agent.pkl", "wb") as f:
            pickle.dump(agent, f)

    return average_q_values, max_q_values, min_q_values, list_cumulative_reaward

if __name__ == "__main__":
    # agent = agents.QLearningAgent()
    agent = agents.RepeatedUpdateQLearningAgent()
    train_agent(agent)