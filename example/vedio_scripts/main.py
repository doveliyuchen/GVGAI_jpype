from agent import Agent
from game import Game

'''
    golddigger:     Window size: 150*270, Tile size:10*10
    treasurekeeper: Window size: 90*130,  Tile size:10*10
    waterpuzzle:    Window size: 110*300, Tile size:10*10
'''

if __name__ == '__main__':
    print('Games:\ngolddigger, treasurekeeper, waterpuzzle\nlevels:lvl0, lvl1')
    gameName = input('game Name:')
    lvlName = input('level name:')
    game = Game(gameName, lvlName)
    game.humanPlay()
    # game.start(Agent())
