#!/usr/bin/env python3
### 
# The following is heavily based on the "example_cartesian_poses_with_notifications.py" file.
# The content of which is copyrighted by Kinova inc. but allowed to be modified.
###

import sys
import rospy
import time
import subprocess
from std_msgs.msg import Int32
from kortex_synergy.msg import CommandActionNotification
from kortex_driver.srv import *
from kortex_driver.msg import *

class CMDCartesianPoseSwitching:
    def __init__(self):
        try:
            rospy.init_node('cmd_cartesian_pose_switching')

            self.HOME_ACTION_IDENTIFIER = 2

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")

            rospy.loginfo("Using robot_name " + self.robot_name)

            # Init the action topic subscriber
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

    def pose_callback(self, data):
        pose_value = data.pose_value
        if pose_value == 1:
            self.send_pose("pose1", 1001, 0.374, 0.081, 0.450, -57.6, 91.1, 2.3)
        elif pose_value == 2:
            self.send_pose("pose2", 1002, 0.374, 0.081, 0.3, -57.6, 91.1, 2.3)
        elif pose_value == 3:
            self.send_pose("pose3", 1003, 0.45, 0.081, 0.3, -57.6, 91.1, 2.3)
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
                time.sleep(0.01)

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def example_home_the_robot(self):
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

    def example_set_cartesian_reference_frame(self):
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

        # Wait a bit
        rospy.sleep(0.25)

    def example_subscribe_to_a_robot_notification(self):
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

        rospy.sleep(1.0)

        return True
    
    def send_pose(self, pose_name, pose_id, x, y, z, theta_x, theta_y, theta_z):
        ## Encapsulation of the pose sending method that was included in the kortex examples.
        
        # Create ActionRequest
        req = ExecuteActionRequest()
        
        # Define pose parameters
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = 0.1  # m/s
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

        # Attemtp to execute received pos
        rospy.loginfo(f"Sending {pose_name}...")
        self.last_action_notif_type = None
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr(f"Failed to send {pose_name}")
            return False
        else:
            rospy.loginfo(f"Waiting for {pose_name} to finish...")
            self.wait_for_action_end_or_abort()
            return True

    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:
            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.example_clear_faults()
            #*******************************************************************************

            #*******************************************************************************
            # Start the example from the Home position
            success &= self.example_home_the_robot()
            #*******************************************************************************
            
            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.example_set_cartesian_reference_frame()

            #*******************************************************************************
            # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
            success &= self.example_subscribe_to_a_robot_notification()

            #*******************************************************************************
            # Start the publisher node as a subprocess
            rospy.loginfo("Starting pose publisher node...")
            try:
                subprocess.Popen(["rosrun", "kortex_synergy", "pose_publisher.py"])
            except OSError:
                rospy.logerr("Failed to start pose publisher node.")
                success = False

            # Subscribe to desired pose topic
            rospy.Subscriber("/"+self.robot_name+"/desired_pose", CommandActionNotification, self.pose_callback)
            
        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")

if __name__ == "__main__":
    cps = CMDCartesianPoseSwitching()
    cps.main()
    rospy.spin()
