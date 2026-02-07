from robot import Robot

STAND_REWARD = 1.0
FALL_PENALTY = -15.0

#needs to be defined

#Max values determined the threshold of how much the robot can tilt before falling
#MAX_ROLL_VAL
#MAX_PITCH_VAL

MAX_STEP_COUNT = 500
class RobotEnv:   
    def __init__(self):
        self.max_steps = MAX_STEP_COUNT
        self.robot = Robot()
        self.step_count = 0
    def reset(self):
        # default robot position
        self.robot = Robot() 
        self.step_count = 0
        return self.robot.get_state()

    def get_state(self):
        return self.robot.get_state()
    
    # need to make sure action is vector of continuous values
    def step(self, action):

        # robot takes action
        self.robot.take_action(action)
        self.step_count += 1

        # get the robot's new state
        state = self.get_state()

        # calculate reward and check done status
        reward = STAND_REWARD
        done = False

        # if robot gets penalty or max steps are taken
        # robot is done with stepoh
        if self.robot.is_fallen():
            reward = FALL_PENALTY
            done = True
        elif (self.step_count >= self.max_steps):
            done = True
        
        return state, reward, done, {}

