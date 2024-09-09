#!/usr/bin/env python3

import rospy
import sys
import os
import select
import tty
import termios

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.pose_listener import ToolPoseListener
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Int32
import random
import time
import csv
from datetime import datetime
import signal
from kortex_driver.srv import *

class Experiment1_singleDoF:
    def __init__(self):
        rospy.init_node('singleDoF_reaching_experiment')
        self.starting_position = (0.45, 0.0, 0.45)
        self.objective_positions = [
            ("Positive X", (round(self.starting_position[0] + 0.4, 3), self.starting_position[1], self.starting_position[2])),  # Positive X
            # ("Negative X", (round(self.starting_position[0] - 0.4, 3), self.starting_position[1], self.starting_position[2])),  # Negative X
            ("Positive Y", (self.starting_position[0], round(self.starting_position[1] + 0.4, 3), self.starting_position[2])),  # Positive Y
            ("Negative Y", (self.starting_position[0], round(self.starting_position[1] - 0.4, 3), self.starting_position[2])),  # Negative Y
            ("Positive Z", (self.starting_position[0], self.starting_position[1], round(self.starting_position[2] + 0.4, 3))),  # Positive Z
            ("Negative Z", (self.starting_position[0], self.starting_position[1], round(self.starting_position[2] - 0.4, 3)))   # Negative Z
        ]
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = "objective_sphere"
        self.model_prefix = "objective_sphere_"
        self.paradigms = [1, 2, 3, 4]
        rospy.loginfo("Waiting for spawn and delete services...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self.spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.load_sphere_model()

        self.paradigm_pub = rospy.Publisher('/selected_paradigm', Int32, queue_size=10)
        self.tool_pose_listener = ToolPoseListener()
        self.collision_margin = 0.08

        # Create a unique filename using the current date and time
        log_path = os.path.join(self.current_dir, r"../logs/")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f'trial_data_{current_time}.csv'
        self.log_file = open(log_path + self.log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['Paradigm', 'Trial', 'Objective', 'Time', 'Success'])
        self.current_trial = None

        # Initialize services for homing the robot
        self.HOME_ACTION_IDENTIFIER = 2
        clear_faults_full_name = '/' + 'my_gen3' + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_full_name)
        self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

        read_action_full_name = '/' + 'my_gen3' + '/base/read_action'
        rospy.wait_for_service(read_action_full_name)
        self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

        execute_action_full_name = '/' + 'my_gen3' + '/base/execute_action'
        rospy.wait_for_service(execute_action_full_name)
        self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

        # Register the signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        # Store original terminal settings for restoring later
        self.original_termios = termios.tcgetattr(sys.stdin)

    def __del__(self):
        self.cleanup()

    def load_sphere_model(self):
        # Define resource path
        path = os.path.join(self.current_dir, r"../gazebo_world/")
        rospy.loginfo("Loading sphere model...")
        try:
            with open(path+"/models/objective.sdf", "r") as f:
                self.sphere_model = f.read()
        except Exception as e:
            rospy.logerr(f"Failed to load sphere model: {e}")
        
    def spawn_objective(self, position, trial):
        model_name = f"{self.model_prefix}{trial}"
        rospy.loginfo(f"Spawning objective {model_name} at position: {position}")
        pose = Pose(Point(*position), Quaternion(0, 0, 0, 0))
        try:
            self.spawn_service(model_name, self.sphere_model, "", pose, "world")
            rospy.sleep(1)  # Wait for Gazebo to spawn the model
        except Exception as e:
            rospy.logerr(f"Failed to spawn objective: {e}")

    def delete_objective(self, trial):
        model_name = f"{self.model_prefix}{trial}"
        rospy.loginfo(f"Deleting objective {model_name}")
        try:
            result = self.delete_service(model_name)
            if not result.success:
                rospy.logwarn(f"Failed to delete objective {model_name}: {result.status_message}")
        except Exception as e:
            rospy.logerr(f"Failed to delete objective: {e}")

    def delete_all_objectives(self):
        for trial in range(len(self.objective_positions) * 10):
            self.delete_objective(trial)

    def check_collision(self, position):
        tool_pose = self.tool_pose_listener.get_tool_pose()
        if all(tool_pose[key] is not None for key in ["x", "y", "z"]):
            return (
                abs(tool_pose["x"] - position[0]) < self.collision_margin and
                abs(tool_pose["y"] - position[1]) < self.collision_margin and
                abs(tool_pose["z"] - position[2]) < self.collision_margin
            )
        return False

    def select_paradigm(self, paradigm):
        ready = input(f"\nSelected paradigm {paradigm}. Ready to start? [y/n]: ")
        if ready.lower() != 'y':
            print("Experiment aborted.")
            sys.exit(0)

        rospy.loginfo(f"Selected paradigm: {paradigm}")
        self.paradigm_pub.publish(paradigm)
        return paradigm

    def run_experiment(self):
        rospy.loginfo("Running experiment...")
        for paradigm in self.paradigms:
            rospy.loginfo(f"Starting trials for paradigm {paradigm}")
            self.print_paradigm_description(paradigm)
            self.select_paradigm(paradigm)
            random.shuffle(self.objective_positions)
            for trial, (name, position) in enumerate(self.objective_positions * 10):  # 10 trials for each position
                rospy.loginfo(f"Trial {trial+1} for position {name}")
                self.spawn_objective(position, trial)
                self.run_trial(name, position, trial, paradigm)
                self.delete_objective(trial)
            self.home_the_robot()

    def print_paradigm_description(self, paradigm):
        descriptions = {
            1: "D2-D4 flexion and extension linked to positive and negative movement along x to z axes respectively.",
            2: "D2-D4 extension and flexion linked to positive and negative movement along x to z axes respectively (inverted of 1).",
            3: "D3-D5 flexion and extension linked to positive and negative movement along x to z axes respectively.",
            4: "D3-D5 extension and flexion linked to positive and negative movement along x to z axes respectively (inverted of 3)."
        }
        rospy.loginfo(descriptions.get(paradigm, "Unknown paradigm"))

    def run_trial(self, name, position, trial, paradigm):
        # Countdown before starting the trial
        for i in range(2, 0, -1):
            rospy.loginfo(f"Starting in {i}...")
            time.sleep(1)
        
        start_time = time.time()
        rospy.loginfo("Go!")
        
        # Maximum time limit for the trial
        max_duration = 60
        end_time = start_time + max_duration

        success = False
        while time.time() < end_time:
            remaining_time = end_time - time.time()
            sys.stdout.write(f"\rTime remaining: {int(remaining_time)} seconds")
            sys.stdout.flush()

            # Check for spacebar press to pause the experiment
            if self.is_key_pressed(' '):
                self.pause_experiment()

            rospy.sleep(1)
            if self.check_collision(position):
                rospy.loginfo("Collision detected!")
                success = True
                break
        
        trial_time = time.time() - start_time
        rospy.loginfo(f"\nTrial completed in: {trial_time:.2f} seconds")
        
        # Log the trial data
        self.csv_writer.writerow([paradigm, trial + 1, name, trial_time, success])
        self.log_file.flush()

    def signal_handler(self, sig, frame):
        print("\nKeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        print("Cleaning up: Sending robot home, deleting all objectives, and closing log file.")
        try:
            self.home_the_robot()
        except Exception as e:
            pass  # Suppress errors during cleanup
        self.delete_all_objectives()
        self.log_file.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_termios)  # Restore terminal settings

    def home_the_robot(self):
        # Clear faults
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")

        # Read the Home Action
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False

        # Execute the Home Action
        req = ExecuteActionRequest()
        req.input = res.output
        rospy.loginfo("Sending the robot home...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteAction")
            return False

        return True

    def pause_experiment(self):
        print("\nExperiment paused.")
        self.home_the_robot()
        input("Robot homed. Press Enter to resume the experiment...")

    def is_key_pressed(self, key):
        # Set terminal to raw mode to capture keypresses
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key_pressed = sys.stdin.read(1) == key
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_termios)  # Restore terminal settings
        return key_pressed

if __name__ == '__main__':
    try:
        experiment = Experiment1_singleDoF()
        experiment.run_experiment()
    except rospy.ROSInterruptException:
        experiment.cleanup()
