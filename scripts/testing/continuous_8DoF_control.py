#!/usr/bin/env python3
### 
# The following is heavily based on the "example_cartesian_poses_with_notifications.py" file.
# The content of which is copyrighted by Kinova inc. but allowed to be modified.
###

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import rospy
import time
import signal
import numpy as np
import threading
from scripts.pose_listener import ToolPoseListener
from scripts.processor_open_ephys import DataProcessor
from std_msgs.msg import Int32
from kortex_synergy.msg import MultiLabelActionNotification
from kortex_driver.srv import *
from kortex_driver.msg import *

class ContinuousControl:
    def __init__(self):
        try:
            rospy.init_node('cmd_cartesian_pose_switching')

            self.HOME_ACTION_IDENTIFIER = 2
            self.pose_count = 0
            
            # Track and define pose (Home = 0,0,0,0,0,0)
            self.pose = {
                'x' : 0,
                'y' : 0,
                'z' : 0,
                'th_x' : 0,
                'th_y' : 0,
                'th_z' : 0
            }

            self.pose_lock = threading.Lock()
            self.stop_event = threading.Event()
            
            # Centralised governance
            self.pose_listener = None
            self.processor = None

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")

            rospy.loginfo("Using robot_name " + self.robot_name)

            # Init the action topic subscriber to handle action progression
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
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
        
        except rospy.ROSException as e:
            rospy.logerr("Failed to initialize: %s", str(e))
            self.is_init_success = False
        
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def get_cartesian_pose(self):    
        # Get cartesian pose from the pose listener    
        response = self.pose_listener.get_tool_pose()
        if response:
            with self.pose_lock:
                # Update global pose 
                self.pose = {
                    "x": response["x"],
                    "y": response["y"],
                    "z": response["z"],
                    "th_x": response["theta_x"],
                    "th_y": response["theta_y"],
                    "th_z": response["theta_z"]
                }
        else:
            rospy.logwarn("Failed to get Cartesian pose")

    def pose_callback(self, data):
        # Count poses:
        self.pose_count += 1

        # Pose value is a binary string of classified multi label representation
        pose_value = np.array([byte for byte in data.pose_array])
        
        # Convert pose_value to command values
        command = [0, 0, 0, 0] # x, y, z, grip

        pos, neg = np.split(pose_value, 2) # split in twain
        command = (command + pos - neg) * 0.01
        
        # Get current position using service
        self.get_cartesian_pose()
        
        # Convert command to pose
        with self.pose_lock:
            self.pose['x'] += command[0]
            self.pose['y'] += command[1]
            self.pose['z'] += command[2]
            #self.pose['th_x']
            #self.pose['th_y']
            #self.pose['th_z']
            act = command[3]

            # Check existence of classified pose and actual pose
            if act is not None: #and cur_pose != None:
                rospy.loginfo(f"Updated desired pose: {self.pose}")
            else:
                rospy.logwarn("Invalid pose value received")

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                pass
                #time.sleep(0.01)
                
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
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
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
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

    def subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
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
        ## Encapsulation of the pose sending method that was included in the kortex examples.
        
        # Create ActionRequest
        req = ExecuteActionRequest()
        
        # Define pose parameters
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

        # Formulate parameters for request
        req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
        req.input.name = pose_name
        req.input.handle.action_type = ActionType.REACH_POSE
        req.input.handle.identifier = pose_id

        # Attempt to execute received pos
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
                self.send_pose("ContinuousPose", self.pose_count, self.pose['x'], self.pose['y'], self.pose['z'], self.pose['th_x'], self.pose['th_y'], self.pose['th_z'])
            time.sleep(0.1)  # Update rate 

    def signal_handler(self, sig, frame):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.stop_event.set()
        if self.processor is not None:
            self.processor.stop()
        rospy.signal_shutdown("KeyboardInterrupt caught")

    def main(self):
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:
            # Register the signal handler for SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self.signal_handler)

            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.clear_robot_faults()
            
            # Start from the Home position
            success &= self.home_the_robot()
            
            # Set the reference frame to "Mixed"
            success &= self.set_reference_frame()

            # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
            success &= self.subscribe_to_a_robot_notification()

            # Start pose listener
            rospy.loginfo("Starting pose listener...")
            self.pose_listener = ToolPoseListener()
            
            # Start the publisher node as a subprocess
            rospy.loginfo("Starting pose publisher...")
            try:
                self.processor = DataProcessor()
                self.processor.start()
            except Exception as e:
                rospy.logerr("Failed to start processor: {}".format(e))
                success = False

            # Subscribe to desired pose topic
            rospy.Subscriber("/"+self.robot_name+"/desired_pose", MultiLabelActionNotification, self.pose_callback)

            # Start the continuous pose update thread
            pose_update_thread = threading.Thread(target=self.continuous_pose_update)
            pose_update_thread.start()

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)

        if not success:
            rospy.logerr("We encountered an error.")
        else:
            rospy.spin()  # Keep the node running


if __name__ == "__main__":
    cps = ContinuousControl()
    cps.main()
    rospy.spin()
