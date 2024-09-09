cmake_minimum_required(VERSION 3.0.2)
project(kortex_synergy)

add_compile_options(-std=c++11)
add_definitions(-D_OS_UNIX)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    rospy 
    std_msgs 
    message_generation 
    actionlib_msgs 
    actionlib 
    kortex_driver
)
find_package(kortex_driver REQUIRED message_generation)


################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  CommandActionNotification.msg
  MultiLabelActionNotification.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  
  kortex_driver
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs message_runtime kortex_driver
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories( include ${catkin_INCLUDE_DIRS})
include_directories(include ${PROJECT_SOURCE_DIR}/src)
include_directories(include ${PROJECT_SOURCE_DIR}/../kortex_api/include/client)
include_directories(include ${PROJECT_SOURCE_DIR}/../kortex_api/include/client_stubs)
include_directories(include ${PROJECT_SOURCE_DIR}/../kortex_api/include/messages)
include_directories(include ${PROJECT_SOURCE_DIR}/../kortex_api/include/common)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   scripts/testing/cartesian_pose_switch_precoded.py
   scripts/testing/cartesian_pose_switch_cmd.py
   scripts/testing/pose_publisher.py
   scripts/testing/cartesian_pose_switch_classifier.py
   scripts/testing/classified_pose_publisher.py
   scripts/testing/multi_label_classifier.py
   scripts/testing/discrete_8DoF_control.py
   scripts/testing/continuous_8DoF_control.py
   scripts/processor_open_ephys.py
   scripts/pose_listener.py
   scripts/ContinuousData.py
   scripts/cartesian_pose_switch_keyboard.py
   scripts/experiment1_keyboard_control.py
   scripts/experiment1_singleDoF.py
   scripts/experiment1_control.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

