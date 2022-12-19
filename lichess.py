import json
import requests
import chess
import random
import pickle

TOKEN = "your_token"

def make_get_requests(req):
    headers = {"Authorization": "Bearer " + TOKEN}
    req = requests.get("https://lichess.org" + req, headers=headers)
    return json.loads(req.text)

def make_post_requests(req):
    headers = {"Authorization": "Bearer " + TOKEN}
    req = requests.post("https://lichess.org" + req, headers=headers)
    return json.loads(req.text)

def make_stream_requests(req):
    headers = {"Authorization": "Bearer " + TOKEN}
    return requests.get("https://lichess.org" + req, headers=headers, stream=True)

# launch once to change normal account to bot account
def upgrade_bot():
    return make_post_requests("/api/bot/account/upgrade")

def get_events():
    return make_stream_requests("/api/stream/event")

def get_game_events(challenge_id):
    return make_stream_requests("/api/bot/game/stream/" + challenge_id)

def accept_challenge(challenge_id):
    return make_post_requests("/api/challenge/" + challenge_id + "/accept")

def play_game(challenge_id):
    # Load the Q-learning agent from a file
    with open("repeated_update_q_learning_agent.pkl", "rb") as f:
        loaded_agent = pickle.load(f)
    for game_request in get_game_events(challenge_id).iter_lines():
        if game_request:
            game = json.loads(game_request)
            # first move, use to know our color
            if game["type"] == "gameFull":
                if game["white"]["name"] == "guicag":
                    my_color = chess.WHITE
                    make_post_requests("/api/bot/game/" + challenge_id + "/move/e2e4")
                else:
                    my_color = chess.BLACK
                print("My color is ", my_color)
            if game["type"] == "gameState":
                board = chess.Board()
                moves = game["moves"]
                for move in moves.split():
                    board.push_uci(move) # synchronize the chess board with the current game
                if board.turn == my_color:
                    move = loaded_agent.get_action(board)
                    print(move)
                    make_post_requests("/api/bot/game/" + challenge_id + "/move/" + str(move))

def main():
    print("BOT guicag is ready -----------------")
    print("Waiting for an event")
    for current_event in get_events().iter_lines():
        if current_event:
            print("Just received an event")
            
            event = json.loads(current_event)
            if event["type"] == "challenge":
                challenge = event["challenge"]
                challenge_id = challenge["id"]
                challenger_name = challenge["challenger"]["name"]
                print("It's challenge from " + challenger_name)
                accept_challenge(challenge_id)
                play_game(challenge_id)

if __name__ == "__main__":
    main()
