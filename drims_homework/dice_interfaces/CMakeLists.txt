cmake_minimum_required(VERSION 3.8)

project(dice_interfaces)

# ================================================================
# BUILD SYSTEM
# ================================================================

find_package(ament_cmake REQUIRED)

# ================================================================
# ROS INTERFACE GENERATION
# ================================================================

find_package(rosidl_default_generators REQUIRED)

# Messages used inside DetectDice.srv
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ================================================================
# GENERATE SERVICE INTERFACE
# ================================================================

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectDice.srv"
  DEPENDENCIES
    sensor_msgs
    geometry_msgs
)

# Required by packages that use the generated interface
ament_export_dependencies(
  rosidl_default_runtime
)

ament_package()