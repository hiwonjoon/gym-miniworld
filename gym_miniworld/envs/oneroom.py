import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces

class OneRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=180,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.box = self.place_entity(Box(color='red'))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class FDOneRoom(MiniWorldEnv):
    def __init__(self, size=4, local_coord=True, **kwargs):
        assert size >= 2
        self.size = size
        self.local_coord = local_coord

        super().__init__(
            max_episode_steps=200,
            **kwargs
        )

        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full((4,), -float('inf')),
            high=np.full((4,), float('inf')),
            dtype=np.float32
        )


    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.place_agent()
        self.agent.radius = 0.2

        self.init_x,_,self.init_y = self.agent.pos

        self.init_pos = np.array([self.init_x,self.init_y])
        self.v = np.array([0.,0.])

    def _move(self):
        next_pos = (self.agent.pos[0]+self.v[0],0.,self.agent.pos[2]+self.v[1])

        if self.intersect(self.agent, next_pos, self.agent.radius):
            self.v = np.array([0.,0.])
        else:
            self.agent.pos = next_pos
            self.agent.dir = np.arctan2(self.v[1],self.v[0])# + 3.1415

    def _get_ob(self):
        ob = np.array([self.agent.pos[0],self.agent.pos[2],self.v[0],self.v[1]],np.float32)
        if self.local_coord:
            ob[:2] -= self.init_pos

        return ob

    def reset(self):
        _ = super().reset()
        return self._get_ob()

    def step(self, action):
        self.step_count += 1

        coeff = 0.05
        self.v = np.clip(self.v + coeff * action,-0.5,0.5)
        self._move()

        if self.step_count >= self.max_episode_steps:
            done = True
        else:
            done = False

        return self._get_ob(), 0.0, done, {}

