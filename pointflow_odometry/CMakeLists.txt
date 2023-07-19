cmake_minimum_required(VERSION 3.0.2)
project(pointflow_odometry)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  pcl_ros
  sensor_msgs
  std_msgs
  tf2
  roscpp
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(OpenMP_LIBRARIES gomp)
endif()


generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pointflow_odometry
  CATKIN_DEPENDS nav_msgs pcl_ros roscpp sensor_msgs std_msgs tf2
  DEPENDS Eigen3 PCL OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/pointflow_odometry_node.cpp src/pfo.cpp src/ieskf.cpp src/util.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenMP_LIBRARIES}
)

set(CMAKE_PREFIX_PATH ${HOME}/smeet/libtorch)
message(STATUS "++++++ torch find path" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (NOT Torch_FOUND)
    message(FATAL_ERROR "you should change CMAKE_PREFIX_PATH to libtorch dir")
else ()
    message(STATUS "found libtorch at " ${TORCH_LIBRARIES})
endif ()

target_link_libraries(${PROJECT_NAME}_node ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME}_node PROPERTY CXX_STANDARD 11)