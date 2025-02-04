"""
helper functions for loading robot urdf
"""

import yaml
import os


def load_robot_config(robot_name=None, config_path=None):
    if config_path is None:
        if robot_name == "franka":
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "franka_config.yaml"
            )
        elif robot_name == "mobile_franka":
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "franka_config.yaml"
            )
        else:
            raise ValueError

    print(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if robot_name is not None:
        assert robot_name == config["robot_name"]
    else:
        robot_name = config["robot_name"]

    return robot_name, config


def get_robot_urdf_path(robot_name):
    print("robot name: {}".format(robot_name))
    if robot_name == "franka":
        urdf_path = os.path.join(os.path.dirname(__file__), "..", "urdf", "panda.urdf")
    elif robot_name == "mobile_franka":
        urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "urdf", "mobilePandaWithGripper.urdf"
        )
    else:
        raise ValueError

    return urdf_path


def get_robot_eef_uid(robot_name):
    if robot_name == "franka":
        eef_uid = 14
    elif robot_name == "mobile_franka":
        eef_uid = 14
    else:
        raise ValueError
    return eef_uid
