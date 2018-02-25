"""Hover task."""

import numpy as np
import math
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

## Copy from TakeOff
class Hover(BaseTask):
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
        self.max_duration = 5.0  # secs
        self.max_error_position = 8.0  # distance units
        self.target_position = np.array([0.0, 0.0, 10.0])  # target position to hover at
        self.weight_position = 1.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 1.5

    def reset(self):
        self.start_z = np.random.normal(20, 4)
        self.start_x = np.random.normal(0 , 4)
        self.start_y = np.random.normal(0 , 4)
        self.last_timestamp = None
        self.last_position = None
        # Nothing to reset; just return initial condition
        p = self.target_position + np.random.normal(0.0, 0.2, size=3)  # slight random position around the target
        return Pose(
                position=Point(*p),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            );

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        position = np.array([pose.position.x, pose.position.y, pose.position.z])

        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)
        state = np.concatenate([position, velocity])

        # Compute reward / penalty and check if this episode is complete
        done = False

        error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
        error_velocity = np.linalg.norm(self.target_velocity - state[3:6])   # Euclidean distance from target velocity vector
        reward = -(self.weight_position * error_position  + self.weight_velocity * error_velocity)

        self.last_timestamp = timestamp
        self.last_position = position

        if error_position > self.max_error_position:
            reward -= 50.0
            print('Last for : {} s'.format(timestamp))
            done = True
        elif timestamp > self.max_duration:
            reward += 50.0  # extra reward, agent made it to the end
            done = True

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
