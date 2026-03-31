#!/usr/bin/env python
import gym_gvgai
import Agent as Agent

games = ['gvgai-testgame1', 'gvgai-testgame2', 'gvgai-testgame3']
validateLevels = ['lvl1-v0', 'lvl2-v0', 'lvl3-v0']
totalTimes = 20

# variables for recording the results
results = {}

for game in games:
    levelRecord = {}
    for level in validateLevels:
        timeRecord = {}
        for t in range(totalTimes):
            env = gym_gvgai.make(game + '-' + level)
            agent = Agent.Agent()
            print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
            stateObs = env.reset()
            actions = env.unwrapped.get_action_meanings()
            totalScore = 0
            for tick in range(2000):
                action_id = agent.act(stateObs, actions)
                stateObs, diffScore, done, debug = env.step(action_id)
                totalScore += diffScore
                print("Action " + str(action_id) + " tick " + str(tick+1) + " reward " + str(diffScore) + " win " + debug["winner"])
                if done:
                    break
            timeRecord[t] = [tick, totalScore, debug["winner"]]
        levelRecord[level] = timeRecord
    results[game] = levelRecord


filename = agent.name + "_result.txt"
with open(filename,'w') as f:
    f.write(str(results))
