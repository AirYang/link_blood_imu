cmake_minimum_required(VERSION 2.4.6)

project (imu)

SET(CMAKE_BUILD_TYPE "Release")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11") 
endif()

add_executable(${CMAKE_PROJECT_NAME} imu.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${CMAKE_PROJECT_NAME} LpSensor.so)
    target_link_libraries(${CMAKE_PROJECT_NAME} pthread)
    target_link_libraries(${CMAKE_PROJECT_NAME} dl)
    target_link_libraries(${CMAKE_PROJECT_NAME} rt)
endif()
