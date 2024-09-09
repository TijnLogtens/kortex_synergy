#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import rospy
import time
import signal
import threading
from scripts.pose_listener import ToolPoseListener
from pygame.locals import *
import pygame
from kortex_driver.srv import *
from kortex_driver.msg import *
from std_msgs.msg import Int32

class Exp1KeyboardControl:
    def __init__(self):
        try:
            rospy.init_node('cmd_cartesian_pose_switching')

            self.HOME_ACTION_IDENTIFIER = 2
            self.pose_count = 0
            
            self.pose = {
                'x' : 0,
                'y' : 0,
                'z' : 0,
                'th_x' : 0,
                'th_y' : 0,
                'th_z' : 0
            }
            self.previous_pose = self.pose.copy()

            self.pose_lock = threading.Lock()
            self.stop_event = threading.Event()
            
            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            rospy.loginfo("Using robot_name " + self.robot_name)

            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

            # Subscriber for the selected paradigm
            self.paradigm = 1
            self.paradigm_sub = rospy.Subscriber('/selected_paradigm', Int32, self.paradigm_callback)
        
        except rospy.ROSException as e:
            rospy.logerr("Failed to initialize: %s", str(e))
            self.is_init_success = False
        
        else:
            self.is_init_success = True

    def paradigm_callback(self, msg):
        self.paradigm = msg.data
        rospy.loginfo(f"Received paradigm: {self.paradigm}")

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def get_cartesian_pose(self):
        response = self.pose_listener.get_tool_pose()
        if response:
            with self.pose_lock:
                self.pose = {
                    "x": response["x"],
                    "y": response["y"],
                    "z": response["z"],
                    "th_x": response["theta_x"],
                    "th_y": response["theta_y"],
                    "th_z": response["theta_z"]
                }
                self.previous_pose = self.pose.copy()
        else:
            rospy.logwarn("Failed to get Cartesian pose")

    def handle_keyboard_input(self):
        pygame.init()
        screen = pygame.display.set_mode((100, 100))

        move_increment = 0.05
        rotate_increment = 1.0

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.stop_event.set()
                    break
                elif event.type == KEYDOWN:
                    self.get_cartesian_pose()
                    with self.pose_lock:
                        if event.key == K_w:
                            self.apply_movement('z', move_increment)
                        elif event.key == K_s:
                            self.apply_movement('z', -move_increment)
                        elif event.key == K_a:
                            self.apply_movement('x', -move_increment)
                        elif event.key == K_d:
                            self.apply_movement('x', move_increment)
                        elif event.key == K_q:
                            self.apply_movement('y', -move_increment)
                        elif event.key == K_e:
                            self.apply_movement('y', move_increment)
                        elif event.key == K_UP:
                            self.pose['th_x'] += rotate_increment
                        elif event.key == K_DOWN:
                            self.pose['th_x'] -= rotate_increment
                        elif event.key == K_LEFT:
                            self.pose['th_y'] -= rotate_increment
                        elif event.key == K_RIGHT:
                            self.pose['th_y'] += rotate_increment
                        elif event.key == K_z:
                            self.pose['th_z'] += rotate_increment
                        elif event.key == K_x:
                            self.pose['th_z'] -= rotate_increment

    def apply_movement(self, axis, increment):
        if self.paradigm == 1:
            self.pose[axis] += increment
        elif self.paradigm == 2:
            self.pose[axis] -= increment
        elif self.paradigm == 3:
            self.pose[axis] += increment
        elif self.paradigm == 4:
            self.pose[axis] -= increment

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False

    def clear_robot_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            return True

    def home_the_robot(self):
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        else:
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()

    def set_reference_frame(self):
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        try:
            self.set_cartesian_reference_frame(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

    def subscribe_to_a_robot_notification(self):
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")
        return True
    
    def send_pose(self, pose_name, pose_id, x, y, z, theta_x, theta_y, theta_z):
        req = ExecuteActionRequest()
        
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = 0.8  # m/s
        my_cartesian_speed.orientation = 15  # deg/s

        my_constrained_pose = ConstrainedPose()
        my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

        my_constrained_pose.target_pose.x = x
        my_constrained_pose.target_pose.y = y
        my_constrained_pose.target_pose.z = z
        my_constrained_pose.target_pose.theta_x = theta_x
        my_constrained_pose.target_pose.theta_y = theta_y
        my_constrained_pose.target_pose.theta_z = theta_z

        req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
        req.input.name = pose_name
        req.input.handle.action_type = ActionType.REACH_POSE
        req.input.handle.identifier = pose_id

        rospy.loginfo(f"Sending {pose_name}...")
        self.last_action_notif_type = None
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr(f"Failed to send {pose_name}")
            return False
        else:
            rospy.loginfo(f"{pose_name} sent successfully")
            return True

    def continuous_pose_update(self):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.pose_lock:
                if self.pose != self.previous_pose:
                    self.send_pose("ContinuousPose", self.pose_count, self.pose['x'], self.pose['y'], self.pose['z'], self.pose['th_x'], self.pose['th_y'], self.pose['th_z'])
                    self.previous_pose = self.pose.copy()
            time.sleep(0.1)

    def signal_handler(self, sig, frame):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.stop_event.set()
        rospy.signal_shutdown("KeyboardInterrupt caught")

    def main(self):
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:
            signal.signal(signal.SIGINT, self.signal_handler)
            success &= self.clear_robot_faults()
            success &= self.home_the_robot()
            success &= self.set_reference_frame()
            success &= self.subscribe_to_a_robot_notification()

            rospy.loginfo("Starting pose listener...")
            self.pose_listener = ToolPoseListener()

            rospy.loginfo("Getting current Cartesian pose after homing...")
            self.get_cartesian_pose()

            rospy.loginfo("Starting continuous pose update...")
            pose_update_thread = threading.Thread(target=self.continuous_pose_update)
            pose_update_thread.start()

            rospy.loginfo("Starting keyboard input handler...")
            keyboard_thread = threading.Thread(target=self.handle_keyboard_input)
            keyboard_thread.start()

        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)

        if not success:
            rospy.logerr("We encountered an error.")
        else:
            rospy.spin()

if __name__ == "__main__":
    kc = Exp1KeyboardControl()
    kc.main()
    rospy.spin()
