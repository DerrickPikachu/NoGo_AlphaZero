cmake_minimum_required(VERSION 3.14)
project(my_project)
find_package(Protobuf REQUIRED)
message("Protobuf_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}")
# include_directories(${Protobuf_INCLUDE_DIRS})
add_subdirectory(proto)

enable_testing()
find_package(GTest REQUIRED)
message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")

# find_package(Torch REQUIRED)

# GoogleTest requires at least C++11

# include(FetchContent)
# FetchContent_Declare(
  #   googletest
  #   URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  # )
  # # For Windows: Prevent overriding the parent project's compiler/linker settings
  # set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # FetchContent_MakeAvailable(googletest)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgtest -pthread")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -lgtest -pthread")

include_directories(game)
include_directories(mock)
include_directories(socket)

add_executable(
  mcts_test
  mcts_test.cc
)

add_executable(
  actor
  actor.cpp
)

add_executable(
  nogo
  game/nogo.cpp
)

target_link_libraries(
  mcts_test
  gtest_main
  gtest
  gmock
  proto
)

target_link_libraries(
  actor
  proto
  # "${TORCH_LIBRARIES}"
)
# set_property(TARGET actor PROPERTY CXX_STANDARD 14)

include(GoogleTest)
gtest_discover_tests(mcts_test)