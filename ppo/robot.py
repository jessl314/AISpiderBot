import numpy as np

# global variables
DEFAULT_SERVO_POSITIONS = np.zeros(12)
#MOVE_FORWARD =
#TURN_LEFT =
#TURN_RIGHT =
# max and min servo positions to ensure safe range of motion
SERVO_MIN = 45
SERVO_MAX = 135
#GAIT CYCLES need to be defined
class Robot:
    def __init__(self):
        # 12 servo positions
        self.servo_positions = DEFAULT_SERVO_POSITIONS
        # 360 degree LiDAR distance readings
        self.lidar_distances = np.zeros(360)

        # self.orientation = np.zeros(3)
        # self.angular_velocity = np.zeros(3)
        
    def take_action(self, choice):
        self.update_servo_position(choice)
    
    def update_servo_position(self, choice):
        # storing normalized choice for next get_state() call
        self.servo_positions = choice

        target_angles = (choice + 1) / 2 * (SERVO_MAX - SERVO_MIN) + SERVO_MIN
        # This is where you would call your PCA9685 library:
        # for i in range(12):
        #     angle = (choice[i] + 1) * 90 # Simple map -1...1 to 0...180
        #     pca.servo[i].angle = angle
    def calculate_reward(self):
        ground_clearance = np.mean(self.lidar_distances)
        if ground_clearance < 0.02:
            return - 10.0
        return 1.0
    def get_state(self):
        # clean-lidar handles out of range LiDAR by replacing 0s with 10.0
        clean_lidar = np.where(self.lidar_distances <= 0, 10.0, self.lidar_distances)
        # brings the lidar into range of 0-1 for the model
        normalized_lidar = np.clip(clean_lidar / 10.0, 0, 1)

        # concatenating servo and lidar into a single state vector for PPO model
        state = np.concatenate([
            self.servo_positions,
            normalized_lidar.flatten()
        ])
        return state.astype(np.float32)
        