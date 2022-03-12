cmake_minimum_required(VERSION 3.14)
project(my_project)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
add_subdirectory(proto)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

enable_testing()
include_directories(game)

add_executable(
  mcts_test
  mcts_test.cc
)

add_executable(
  actor
  actor.cpp
)

target_link_libraries(
  mcts_test
  gtest_main
)

target_link_libraries(
  actor proto
)

include(GoogleTest)
gtest_discover_tests(mcts_test)