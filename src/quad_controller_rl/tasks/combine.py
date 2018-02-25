"""Hover task."""
import numpy as np
import math
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combine(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""
    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 3000.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        self.state_size = 10
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]
        self.last_action = np.zeros(6);
        # Task-specific parameters
        self.max_duration = 7.0  # secs
        self.max_error_velocity = 13.0  # speed units
        self.max_error_position = 5.0 # distance units xy
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to land at
        self.weight_position_xy = 0.5
        self.weight_position_z = 0.9
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.3
        self.target_velocity = np.array([0.0, 0.0, -2.0])  # target velocity
        self.weight_velocity = 0.7

    def reset(self):
        self.task = "Takeoff"
        self.last_timestamp = None
        self.last_position = None
        self.takeoff_pass = False
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            );

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)

        done = False
        task = self.task
        if(self.task == "Takeoff"):
            self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at
            self.weight_position = 1.5
            self.target_velocity = np.array([0.0, 0.0, 2.0])  # target velocity (ideally 2 units per sec)
            self.weight_velocity = 0.6
            task = 1
        elif(self.task == "Hover"):

            self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at
            self.weight_position = 1.5
            self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
            self.weight_velocity = 0.8
            task = 2
        else:
            self.target_position = np.array([0.0, 0.0, 0.0])  # target position to land at
            self.weight_position_xy = 0.5
            self.weight_position_z = 1.8
            self.target_velocity = np.array([0.0, 0.0, -2.0])  # target velocity
            self.weight_velocity = 0.7
            task = 3
        state = np.concatenate([position, velocity, [task]])


        if(self.task == "Takeoff"):
            error_position = np.linalg.norm(self.target_position - state[0:3])
            error_velocity = np.linalg.norm(self.target_velocity - state[3:6])   # Euclidean distance from target velocity vector

            reward = -(self.weight_position * error_position  + self.weight_velocity * error_velocity)

            if (timestamp > 5.0):
                reward -= 50.0
                print('Failed to takeoff')
                done = True
            elif (pose.position.z >= 10.0):
                reward += 50.0
                print("Task is Hover")
                self.task = "Hover"


        elif(self.task == "Hover"):
            error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
            # error_orientation = np.linalg.norm(self.target_orientation - state[3:6])  # Euclidean distance from target orientation quaternion (a better comparison may be needed)
            error_velocity = np.linalg.norm(self.target_velocity - state[3:6])   # Euclidean distance from target velocity vector
            reward = -(self.weight_position * error_position  + self.weight_velocity * error_velocity)
            if state[2] < 0.3:
                print('Clash Last for : {} s'.format(timestamp))
                reward -= 100.0 # penalty clash
                done = True
            elif timestamp > 10:
                reward += 50;
                self.task = "Landing"
                print("Task is Landing")

        elif(self.task == "Landing"):
            error_position_xy = np.linalg.norm(self.target_position[0:2] - state[0:2])
            error_position_z = np.linalg.norm(self.target_position[2] - state[2]) #

            error_velocity = np.linalg.norm(self.target_velocity - state[3:6])
            if (state[2] < 8.0):    # less than 8 units ... slower please
                reward = -(self.weight_position_z * error_position_z + self.weight_position_xy * error_position_xy + 2.0 * error_velocity * self.weight_velocity )
            else:
                reward = -(self.weight_position_z * error_position_z + self.weight_position_xy * error_position_xy + self.weight_velocity * error_velocity)
            if timestamp > 20:
                print('Too long to land, Last for : {} s'.format(timestamp))
                reward -= 200.0
                done = True
            elif state[2] < +0.3:
                print("Landed for {} s".format(timestamp))
                reward += 3000 # We land yeh!
                done = True

        self.last_timestamp = timestamp
        self.last_position = position
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            self.last_action = action
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
