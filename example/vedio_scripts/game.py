import pygame as pg
import gym_gvgai as gvg


class Game:
    def __init__(self, game, lvl):
        self.env = gvg.make('gvgai-' + game + '-' + lvl + '-v0')
        self.stateObs = self.env.reset()
        size = (len(self.stateObs), len(self.stateObs[0]))
        self.transpose = size[0] < size[1]
        if self.transpose:
            self.size = (size[1]*2, size[0]*2)
        else:
            self.size = (size[0]*2, size[1]*2)
        self.done = False
        self.score = 0
        self.frame = 0
        self.nAction = self.env.action_space.n

    def start(self, agent, maxT=1000, printLog=True, visualized=True, fps=10):
        if visualized:
            clk = pg.time.Clock()
            screen = pg.display.set_mode(self.size)
            for i in range(maxT):
                clk.tick(fps)
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                self.update(agent, printLog)
                self.draw(screen)
                pg.display.flip()
                if self.done:
                    print('---------------------------\nFinish. Final score = %d' % self.score)
                    return
        else:
            for i in range(maxT):
                self.update(agent, printLog)
                if self.done:
                    print('---------------------------\nFinish. Final score = %d' % self.score)
                    return

    def humanPlay(self):
        print('Use direction keys to move, z key to take other actions(if exist in this game).')
        screen = pg.display.set_mode(self.size)
        while not self.done:
            evt = pg.event.wait()
            if evt.type == pg.QUIT:
                pg.quit()
                self.done = True
            elif evt.type == 3:
                self.playerAct(self.parseKey(evt))
            if self.done:
                print('---------------------------\nFinish. Final score = %d' % self.score)
                return
            self.draw(screen)
            pg.display.flip()

    def parseKey(self, evt):
        if evt.key == pg.K_z:
            if self.nAction > 5:
                return 1
            else:
                return 0
        if evt.key == pg.K_x:
            if self.nAction > 6:
                return 2
            else:
                return 0
        elif evt.key == pg.K_UP:
            return self.nAction-1
        elif evt.key == pg.K_DOWN:
            return self.nAction-2
        elif evt.key == pg.K_RIGHT:
            return self.nAction - 3
        elif evt.key == pg.K_LEFT:
            return self.nAction - 4
        else:
            return 0

    def playerAct(self, actionID):
        self.stateObs, reward, self.done, debug = self.env.step(actionID)
        self.score += reward
        self.frame += 1
        print('frame%d, action:%d, reward:%d, score:%d' % (self.frame, actionID, reward, self.score))

    def update(self, agent, printLog=True):
        action_id = agent.act(self.stateObs, self.env.action_space)
        self.stateObs, reward, self.done, debug = self.env.step(action_id)
        self.score += reward
        self.frame += 1
        if printLog:
            print('frame%d, action:%d, reward:%d, score:%d' % (self.frame, action_id, reward, self.score))

    def draw(self, screen):
        buffer = pg.pixelcopy.make_surface(self.stateObs[:, :, :3])
        pa = pg.PixelArray(buffer)
        if self.transpose:
            pa = pa.transpose()
        screen.blit(pg.transform.scale(pa.make_surface(), self.size), (0, 0))
