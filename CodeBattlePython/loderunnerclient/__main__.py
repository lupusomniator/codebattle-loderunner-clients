import logging
from loderunnerclient.game_client import GameClient
from loderunnerclient.environment import Environment



logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

def turn(board):
    board.save_to_file("last_board")

def server_main():
    env = Environment()
    gcb = GameClient(
        # change this url to your
        "https://dojorena.io/codenjoy-contest/board/player/dojorena392?code=407418550408423703"
    )
    gcb.run(env.on_turn)

def local_main():
    env = Environment()
    board = Board.load_from_file("last_board")
    game = Game(board)
    while True:
        action = env.on_turn(game.to_string_board(), reward)
        reward = game.apply_to_main_hero(action)

if __name__ == "__main__":
    # Board.load_from_file("last_board")
    # for i in range(10000):
        # turn(Board.load_from_file("last_board"))
    #main()
    local_main()
