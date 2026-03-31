"""
This is a Template of Agent file. 
Your agent must contains the Agent class, the act function and the Agent's name. 
"""

from random import randint

class Agent():

    def __init__(self):
        self.name = "randomAgent"

    def act(self, stateObs, actions):
        ### Your algorithm and policy
        action_id = randint(0,len(actions)-1)
        return action_id
