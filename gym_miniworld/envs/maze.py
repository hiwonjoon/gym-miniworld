import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS

class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        max_episode_steps=None,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_maze(self,maze_id=None):
        np_random = np.random.RandomState()
        np_random.seed(maze_id)

        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    #floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]
            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = np_random.permutation([(0,1), (0,-1), (-1,0), (1,0)])

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(np_random.randint(self.num_rows),np_random.randint(0,self.num_cols))

    def _gen_world(self):
        self._gen_maze()

        self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class MazeS2(Maze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)

class MazeS3(Maze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)

class MazeS3Fast(Maze):
    def __init__(self, forward_step=0.7, turn_step=45):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = 300

        super().__init__(
            num_rows=3,
            num_cols=3,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False
        )


class FDMaze(Maze):
    def __init__(self, num_rows=2, num_cols=2, room_size=1, local_coord=False, **kwargs):
        self.local_coord = local_coord
        self._maze_id = None # changes everytime when this env is reset.

        super(FDMaze, self).__init__(
            num_rows,num_cols,room_size,
            max_episode_steps=200,
            window_width=600,
            window_height=600,
            **kwargs
        )

        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full((4,), -float('inf')),
            high=np.full((4,), float('inf')),
            dtype=np.float32
        )

    def _gen_world(self):
        self._gen_maze(self._maze_id)

        self.place_agent()
        self.agent.radius = 0.1

        self.init_x,_,self.init_y = self.agent.pos

        self.init_pos = np.array([self.init_x,self.init_y])
        self.v = np.array([0.,0.])

    def inside(self,pos):
        for r in self.rooms:
            if r.point_inside(pos):
                return True
        return False

    def _move(self):
        next_pos = (self.agent.pos[0]+self.v[0],0.,self.agent.pos[2]+self.v[1])

        if self.intersect(self.agent, next_pos, self.agent.radius):
            self.v = np.array([0.,0.])
        elif not self.inside(next_pos):
            self.v = np.array([0.,0.])
        else:
            self.agent.pos = next_pos
            self.agent.dir = np.arctan2(self.v[1],self.v[0])# + 3.1415

    def _get_ob(self):
        ob = np.array([self.agent.pos[0],self.agent.pos[2],self.v[0],self.v[1]],np.float32)
        if self.local_coord:
            ob[:2] -= self.init_pos

        return ob

    def reset(self,maze_id=None):
        self._maze_id = maze_id
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


