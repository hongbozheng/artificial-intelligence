import numpy as np
import utils

# Reward Direction Const
SAME_X_COORD,SAME_Y_COORD = 0,0
REWARD_ON_LEFT,REWARD_ON_TOP = 1,1
REWARD_ON_RIGHT,REWARD_ON_BOTTOM = 2,2

# Wall Direction Const
NO_X_ADJ_WALL,NO_Y_ADJ_WALL = 0,0
WALL_ON_LEFT,WALL_ON_TOP = 1,1
WALL_ON_RIGHT,WALL_ON_BOTTOM = 2,2

# Snake Direction Const
NO_SNAKE_BODY = 0
SNAKE_BODY_ON_TOP,SNAKE_BODY_ON_BOTTOM,SNAKE_BODY_ON_LEFT,SNAKE_BODY_ON_RIGHT = 1,1,1,1

# Playable Space Range Const
PLAYABLE_SPACE_X_MIN,PLAYABLE_SPACE_Y_MIN = utils.GRID_SIZE,utils.GRID_SIZE
PLAYABLE_SPACE_X_MAX,PLAYABLE_SPACE_Y_MAX = utils.DISPLAY_SIZE-2*utils.GRID_SIZE,utils.DISPLAY_SIZE-2*utils.GRID_SIZE

class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne        # used in exploration function
        self.C = C
        self.gamma = gamma  # discount factor
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        # (reward_dir_x,reward_dir_y,adj_wall_x,adj_wall_y,adj_body_top,adj_body_bottom,adj_body_left,adj_body_right)
        s_prime = self.generate_state(environment)
        # TODO: write your function here

        if self._train == True and self.s is not None and self.a is not None:
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1
            else:
                reward = -0.1
            self.updateNTable()
            self.updateQTable(s_prime,reward)

        if dead == False:
            self.s = s_prime
            self.points = points
        else:
            self.reset()
            return 0

        self.a = self.getOptimalAction(self.s)
        return self.a

        #return NULL

    def getOptimalAction(self,s_prime):
        """
        function to return the optimal action for agent
        :param s_prime: tuple of next state info
        :param action: action of agent
        :return: optimal action of agent
        """
        # f_QN: f(Q(s,a),N(s,a)) = 1      --- N(s,a) < Ne
        # f_QN: f(Q(s,a),N(s,a)) = Q(s,a) --- else
        optimal_action = 0
        optimal_f_QN = -0xFFFFFFFF
        for action in self.actions:
            if self.N[s_prime][action] < self.Ne:
                f_QN = 1
            else:
                f_QN = self.Q[s_prime][action]
            if f_QN >= optimal_f_QN:
                optimal_f_QN = f_QN
                optimal_action = action
        return optimal_action

    def getQMax(self,s_prime):
        Q_max = -0xFFFFFFFF
        for action in self.actions:
            if Q_max < self.Q[s_prime][action]:
                Q_max = self.Q[s_prime][action]
        return Q_max

    def updateQTable(self,s_prime,reward):
        lrate = self.C/(self.C+self.N[self.s][self.a])
        Q_max = self.getQMax(s_prime)
        self.Q[self.s][self.a] = self.Q[self.s][self.a]+lrate*(reward+self.gamma*Q_max-self.Q[self.s][self.a])

    def updateNTable(self):
        self.N[self.s][self.a] += 1

    def generate_state(self,environment):
        """
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        :return: a tuple of
        (reward_dir_x, reward_dir_y, adj_wall_x, adj_wall_y, adj_body_top, adj_body_bottom, adj_body_left, adj_body_right)
        """
        # TODO: Implement this helper function that generates a state given an environment
        # snake_loc = (environment[0],environment[1])
        # snake_body = environment[2]
        # reward_loc = (environment[3],environment[4])

        return (self.checkRewardDirX((environment[0],environment[1]),(environment[3],environment[4])),
                self.checkRewardDirY((environment[0],environment[1]),(environment[3],environment[4])),
                self.checkAdjWallX((environment[0],environment[1])),
                self.checkAdjWallY((environment[0],environment[1])),
                self.checkAdjBodyTop((environment[0],environment[1]),environment[2]),
                self.checkAdjBodyBottom((environment[0],environment[1]),environment[2]),
                self.checkAdjBodyLeft((environment[0],environment[1]),environment[2]),
                self.checkAdjBodyRight((environment[0],environment[1]),environment[2]))


    def checkRewardDirX(self,snake_loc,reward_loc):
        if snake_loc[0] == reward_loc[0]:
            return SAME_X_COORD
        elif snake_loc[0] > reward_loc[0]:
            return REWARD_ON_LEFT
        else:
            return REWARD_ON_RIGHT

    def checkRewardDirY(self,snake_loc,reward_loc):
        if snake_loc[1] == reward_loc[1]:
            return SAME_Y_COORD
        elif snake_loc[1] > reward_loc[1]:
            return REWARD_ON_TOP
        else:
            return REWARD_ON_BOTTOM

    def checkAdjWallX(self,snake_loc):
        if snake_loc[0] > PLAYABLE_SPACE_X_MIN and snake_loc[0] < PLAYABLE_SPACE_X_MAX:
            return NO_X_ADJ_WALL
        elif snake_loc[0] <= PLAYABLE_SPACE_X_MIN:
            return WALL_ON_LEFT
        else:
            return WALL_ON_RIGHT

    def checkAdjWallY(self,snake_loc):
        if snake_loc[1] > PLAYABLE_SPACE_Y_MIN and snake_loc[1] < PLAYABLE_SPACE_Y_MAX:
            return NO_Y_ADJ_WALL
        elif snake_loc[1] <= PLAYABLE_SPACE_Y_MIN:
            return WALL_ON_TOP
        else:
            return WALL_ON_BOTTOM

    def checkAdjBodyTop(self,snake_loc,snake_body):
        if not snake_body:
            return NO_SNAKE_BODY
        else:
            for coord in snake_body:
                if snake_loc[0] == coord[0] and (snake_loc[1]-utils.GRID_SIZE) == coord[1]:
                    return SNAKE_BODY_ON_TOP
            return NO_SNAKE_BODY

    def checkAdjBodyBottom(self,snake_loc,snake_body):
        if not snake_body:
            return NO_SNAKE_BODY
        else:
            for coord in snake_body:
                if snake_loc[0] == coord[0] and (snake_loc[1]+utils.GRID_SIZE) == coord[1]:
                    return SNAKE_BODY_ON_BOTTOM
            return NO_SNAKE_BODY

    def checkAdjBodyLeft(self,snake_loc,snake_body):
        if not snake_body:
            return NO_SNAKE_BODY
        else:
            for coord in snake_body:
                if (snake_loc[0]-utils.GRID_SIZE) == coord[0] and snake_loc[1] == coord[1]:
                    return SNAKE_BODY_ON_LEFT
            return NO_SNAKE_BODY

    def checkAdjBodyRight(self,snake_loc,snake_body):
        if not snake_body:
            return NO_SNAKE_BODY
        else:
            for coord in snake_body:
                if (snake_loc[0]+utils.GRID_SIZE) == coord[0] and snake_loc[1] == coord[1]:
                    return SNAKE_BODY_ON_RIGHT
            return NO_SNAKE_BODY
