# wzhi: bullet functions and helpers
import numpy as np
import torch
from torch import nn
from pyquaternion import Quaternion

import pybullet as p
from envs import FrankaEnv
from envs.robot_env import RobotEnv
from envs import robot_sim
from utils.python_utils import merge_dicts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "initial_goal_distance_min": 0.5,
}


class cylinder:
    def __init__(self, pos, orientation, lengths):
        self.type = "cylinder"
        self.pos = pos
        self.orientation = orientation
        self.r = lengths[1]
        self.h = lengths[0]

    def generate_samples(self, nn=300):
        samp_angle = np.random.uniform(0, 2 * np.pi, size=(nn, 1))
        samp_r = np.random.uniform(0, self.r, size=(nn, 1))
        samp_h = np.random.uniform(-self.h / 2, self.h / 2, size=(nn, 1))
        samp_x = samp_r * np.cos(samp_angle)
        samp_y = samp_r * np.sin(samp_angle)
        samp_xyz_origin = np.hstack([samp_x, samp_y, samp_h])
        cur_rot = Quaternion(axis=self.orientation[:3], angle=self.orientation[3])
        # R = cur_rot.rotation_matrix
        # rot_samps=np.matmul(samp_xyz_origin, np.transpose(R))
        rot_samps_l = []
        for i in range(len(samp_xyz_origin)):
            rot_point = cur_rot.rotate(samp_xyz_origin[i])
            rot_samps_l.append(rot_point.copy())
        rot_samps_np = np.array(rot_samps_l).reshape((-1, 3))
        pos_rot = self.pos  # cur_rot.rotate(self.pos)
        samp_xyz = rot_samps_np + pos_rot
        return samp_xyz


class cube:
    def __init__(self, pos, orientation, lengths):
        self.type = "cube"
        self.pos = np.array(pos)
        self.orientation = np.array(orientation)
        self.lengths = np.array(lengths)

    def generate_samples(self, nn=300):
        samp_xyz_origin = np.random.uniform(
            -self.lengths / 2, self.lengths / 2, size=(nn, 3)
        )
        samp_xyz_origin_ext = np.concatenate(
            [samp_xyz_origin, np.ones((len(samp_xyz_origin), 1))], axis=1
        )
        cur_rot = Quaternion(
            w=self.orientation[3],
            x=self.orientation[0],
            y=self.orientation[1],
            z=self.orientation[2],
        )
        R = cur_rot.rotation_matrix
        T = self.pos.reshape((3, 1))
        Z = np.zeros((1, 4))
        T_mat_part = np.hstack([R, T])
        T_mat = np.vstack([T_mat_part, Z])
        T_mat[3, 3] = 1.0
        rot_samps = np.matmul(T_mat, samp_xyz_origin_ext.T)
        samp_xyz = rot_samps.T[:, :3]  # rot_samps_np+self.pos
        return samp_xyz


class CleanFrankaEnv(RobotEnv):
    """
    Base clean gym environment for franka robot
    """

    def __init__(self, mobile=False, config=None):
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()
        print(mobile)
        if mobile == False:
            robot_name = "franka"
        else:
            robot_name = "mobile_franka"
        print(robot_name)
        super().__init__(robot_name=robot_name, workspace_dim=3, config=config)

    def simple_reset(self):
        """
        reset time and simulator
        """
        # reset time and simulator
        self.terminated = False
        self._env_step_counter = 0
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._time_step)
        self._p.setGravity(0, 0, self._gravity)

        self.goal_uid = None
        self.obs_uids = []

        # create robot
        self._robot = robot_sim.create_robot_sim(
            self.robot_name, self._p, self._time_step, mode=self._acc_control_mode
        )


def add_wp(bullet_client, position, radius=0.01, color=[1.0, 0.0, 0.0, 1]):
    """
    Adds a red waypoint
    """
    collision = -1
    visual = bullet_client.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=color
    )
    goal = bullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    return goal


def setup_env_bullet(world_obj_list):
    id_list = []
    for i in range(len(world_obj_list)):
        if world_obj_list[i].type == "cylinder":
            radius = world_obj_list[i].r
            height = world_obj_list[i].h
            c_pos = world_obj_list[i].pos
            c_ori = world_obj_list[i].orientation
            # create blue cylinders
            color = [0.0, 0.0, 1.0, 1.0]
            visual = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color
            )
            obj = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=c_pos,
                baseOrientation=c_ori,
            )
            id_list.append(obj)
        if world_obj_list[i].type == "cube":
            test_len = world_obj_list[i].lengths / 2
            c_pos = world_obj_list[i].pos
            c_ori = world_obj_list[i].orientation
            color = [np.random.rand(1), np.random.rand(1), np.random.rand(1), 1]
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=test_len, rgbaColor=color
            )
            obj = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=c_pos,
                baseOrientation=c_ori,
            )
            id_list.append(obj)
    return id_list
