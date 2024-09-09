#!/usr/bin/env python

import rospy
from kortex_driver.msg import *

class ToolPoseListener:
    def __init__(self):
        self.tool_pose_x = None
        self.tool_pose_y = None
        self.tool_pose_z = None
        self.tool_pose_theta_x = None
        self.tool_pose_theta_y = None
        self.tool_pose_theta_z = None

        rospy.Subscriber('/my_gen3/base_feedback', BaseCyclic_Feedback, self.callback)
    
    def callback(self, data):
        self.tool_pose_x = data.base.tool_pose_x
        self.tool_pose_y = data.base.tool_pose_y
        self.tool_pose_z = data.base.tool_pose_z
        self.tool_pose_theta_x = data.base.tool_pose_theta_x
        self.tool_pose_theta_y = data.base.tool_pose_theta_y
        self.tool_pose_theta_z = data.base.tool_pose_theta_z

    def get_tool_pose(self):
        return {
            "x": self.tool_pose_x,
            "y": self.tool_pose_y,
            "z": self.tool_pose_z,
            "theta_x": self.tool_pose_theta_x,
            "theta_y": self.tool_pose_theta_y,
            "theta_z": self.tool_pose_theta_z
        }
