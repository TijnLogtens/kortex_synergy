#!/usr/bin/env python

import rospy
from kortex_synergy.msg import CommandActionNotification

def main():
    rospy.init_node('pose_publisher')
    # Get node params
    robot_name = rospy.get_param('~robot_name', "my_gen3")

    pub = rospy.Publisher('/'+robot_name+'/desired_pose', CommandActionNotification, queue_size=10)
    rate = rospy.Rate(10)  # Adjust the publishing rate as needed

    while not rospy.is_shutdown():
        pose_value = input("Enter pose value (1, 2, or 3) or 'q' to quit: ")
        
        if pose_value.lower() == 'q':
            rospy.loginfo("Exiting pose publisher.")
            break
        
        try:
            pose_value = int(pose_value)
            if pose_value in [1, 2, 3]:
                msg = CommandActionNotification()
                msg.pose_value = pose_value
                pub.publish(msg)
                rospy.loginfo("Published pose value: %d", pose_value)
            else:
                rospy.logwarn("Invalid pose value. Please enter 1, 2, or 3.")
        except ValueError:
            rospy.logwarn("Invalid input. Please enter a number or 'q' to quit.")

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
