#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <errno.h>

#include "mock_class.h"
#include "../proto/trajectory.pb.h"
#include "trajectory_socket.h"
using ::testing::AtLeast;
using ::testing::ReturnRef;
using ::testing::Return;


// Demonstrate some basic assertions.
TEST(FrameworkTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

/*********************************************************************
* SelfPlayEngineTest
* In this section, all the test cases are written to test the class 
* SelfPlayEngineTest.
* SelfPlayEngine should do the self play and generate the trajectory
* which is used to train an Neural Network(AlphaZero Resnet)
**********************************************************************/
class SelfPlayEngineTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string player_arg = 
        "simulation=400 explore=0.3 uct=normal parallel=1";
    black_ = new player("name=mcts " + player_arg + " role=black");
    white_ = new player("name=mcts " + player_arg + " role=white");
    engine = new SelfPlayEngine(black_, white_);
  }

  void TearDown() override {
    delete engine;
    delete black_;
    delete white_;
  }

  SelfPlayEngine* engine;
  player* black_;
  player* white_;
};

TEST_F(SelfPlayEngineTest, OpenEpisodeTest) {
  // Test the episode.open_episode() has been called
  EpisodeMock episode_mock;
  std::string episode_str = black_->name() + ":" + white_->name();
  board dummy_board;
  EXPECT_CALL(episode_mock, open_episode(episode_str))
    .Times(1);
  EXPECT_CALL(episode_mock, state())
    .WillOnce(ReturnRef(dummy_board));
  engine->init_game(&episode_mock);
  engine->open_episode();

  trajectory answer;
  trajectory::transition* answer_transition = answer.add_transitions();
  auto* answer_state = answer_transition->mutable_state();
  for (int x = 0; x < board::size_x; x++) {
    for (int y = 0; y < board::size_y; y++) {
      answer_state->Add(0.0);
    }
  }
  std::string answer_str;
  answer.SerializeToString(&answer_str);
  EXPECT_EQ(engine->get_trajectory(), answer_str);
}

TEST_F(SelfPlayEngineTest, CloseEpisodeTest) {
  // Test the close_episode()
  /*
  * This function should identify the winner, 
  * and call the close episode of episode.
  * Additionally, it also needs to setup the last transition reward.
  */
  EpisodeMock episode_mock;
  agent dummy_winner("name=test role=white");
  board dummy_board;
  std::string episode_str = black_->name() + ":" + white_->name();
  EXPECT_CALL(episode_mock, last_turns(*black_, *white_))
    .WillOnce(ReturnRef(dummy_winner));
  EXPECT_CALL(episode_mock, close_episode(dummy_winner.role()))
    .Times(1);
  EXPECT_CALL(episode_mock, open_episode(episode_str));
  EXPECT_CALL(episode_mock, state())
    .WillOnce(ReturnRef(dummy_board));
  engine->init_game(&episode_mock);
  engine->open_episode();
  agent& winner = engine->close_episode();

  std::string winner_name = winner.name();
  std::string winner_name_ = dummy_winner.name();
  EXPECT_EQ(winner_name, winner_name_);

  std::string raw = engine->get_trajectory();
  trajectory parsed_raw;
  ASSERT_EQ(parsed_raw.ParseFromString(raw), true);
  int num_transition = parsed_raw.transitions_size();
  ASSERT_EQ(num_transition, 1);
  trajectory::transition* transition = 
    parsed_raw.mutable_transitions(num_transition - 1);
  EXPECT_EQ(transition->reward(), -1.0);
}

TEST_F(SelfPlayEngineTest, NextActionTest) {
  // Test do_next_action()
  /*
  * This function should choose the correct player to play the game,
  * and return the action choosen by player
  */
  EpisodeMock episode_mock;
  AgentMock player_mock;
  action::place dummy_move(0, board::black);
  board dummy_board;
  EXPECT_CALL(episode_mock, take_turns(*black_, *white_))
    .WillOnce(ReturnRef(player_mock));
  EXPECT_CALL(episode_mock, state())
    .WillOnce(ReturnRef(dummy_board));
  EXPECT_CALL(player_mock, take_action(dummy_board))
    .WillOnce(Return(dummy_move));
  engine->init_game(&episode_mock);
  action choosen_action = engine->next_action();
  EXPECT_EQ(std::string(((action::place)choosen_action).position()),
    std::string(dummy_move.position()));
}

TEST_F(SelfPlayEngineTest, applyLegalActionTest) {
  // Test apply_action()
  /*
  * Try to apply a legal action to the game
  */
  action::place dummy_move(0, board::black);
  EpisodeMock episode_mock;
  EXPECT_CALL(episode_mock, apply_action(dummy_move))
    .WillOnce(Return(true));
  engine->init_game(&episode_mock);
  bool result = engine->apply_action(dummy_move);
  EXPECT_EQ(result, true);
}

TEST_F(SelfPlayEngineTest, applyIllegalActionTest) {
  // Test apply_action()
  /*
  * Try to apply a illegal action to the game
  */
  action::place dummy_move(0, board::black);
  EpisodeMock episode_mock;
  EXPECT_CALL(episode_mock, apply_action(dummy_move))
    .WillOnce(Return(false));
  engine->init_game(&episode_mock);
  bool result = engine->apply_action(dummy_move);
  EXPECT_EQ(result, false);
}

TEST_F(SelfPlayEngineTest, storeTransitionTest) {
  // Test store_transition()
  /*
  * Check that after applying
  */
  trajectory answer;
  for (int i = 0; i < 2; i++) {
    trajectory::transition* answer_transition = answer.add_transitions();
    auto* answer_state = answer_transition->mutable_state();
    for (int x = 0; x < board::size_x; x++) {
      for (int y = 0; y < board::size_y; y++) {
        answer_state->Add(0.0);
      }
    }
  }
  auto* first_transition = answer.mutable_transitions(0);
  first_transition->set_reward(0);
  first_transition->set_action_id(1);
  auto* second_transition = answer.mutable_transitions(1);
  float* value = second_transition->mutable_state()->Mutable(1);
  *value = 1.0;
  std::string answer_str;
  answer.SerializeToString(&answer_str);

  action::place dummy_move(1, board::black);
  EpisodeInterface* real_game = new episode();
  
  engine->init_game(real_game);
  engine->open_episode();
  engine->apply_action(dummy_move);
  engine->store_transition(dummy_move);
  EXPECT_EQ(engine->get_trajectory(), answer_str);
}

TEST_F(SelfPlayEngineTest, fullTrajectoryTest) {
  // Test trajectory generated by self play
  /*
  * Check the correctness of the generated trajectory raw data.
  */
  std::vector<action> action_record;
  agent* dummy_black = new player("name=dummy_black role=black");
  agent* dummy_white = new player("name=dummy_white role=white");
  delete engine;
  engine = new SelfPlayEngine(dummy_black, dummy_white);
  EpisodeInterface* game = new episode();
  engine->init_game(game);
  engine->open_episode();
  while (true) {
    action move = engine->next_action();
    if (engine->apply_action(move) == false) break;
    engine->store_transition(move);
    action_record.push_back(move);
  }
  agent& winner = engine->close_episode();
  std::string raw = ((SelfPlayEngine*)engine)->get_trajectory();
  
  trajectory tem_trajectory;
  tem_trajectory.ParseFromString(raw);
  auto* transitions = tem_trajectory.mutable_transitions();
  for (int i = 0; i < action_record.size(); i++) {
    board::point point = ((action::place)action_record[i]).position();
    int action_id = point.x * board::size_y + point.y;
    EXPECT_EQ(action_id, transitions->at(i).action_id());
  }
  EXPECT_NE(tem_trajectory.transitions_size(), 0);
} // End of SelfPlayEngineTest

/*********************************************************************
* TrajectorySocketTest
* The class TrajectorySocket must connect to the learner, and has the
* ability of sending an serilized trajectory(protobuf) to learner.
**********************************************************************/
class TrajectorySocketTest : public ::testing::Test {
protected:
  class TrajectorySocketStub : public TrajectorySocket {
  public:
      TrajectorySocketStub(int pipe_write_fd)
          : TrajectorySocket(), write_fd(pipe_write_fd) {}

      bool connect_server() override {
          socket_fd = write_fd;
          return true;
      }

      int write_fd;
  };

protected:
  void SetUp() {
    pipe(test_pipe);
    test_socket = new TrajectorySocketStub(test_pipe[1]);
  }

  void TearDown() {
    delete test_socket;
    close(test_pipe[0]);
    close(test_pipe[1]);
  }

  SocketInterface* test_socket;
  int test_pipe[2];
};

TEST_F(TrajectorySocketTest, sendTest) {
  trajectory test_trajectory;
  auto* transition = test_trajectory.add_transitions();
  transition->set_action_id(10);
  transition->set_reward(100);
  std::string raw;
  test_trajectory.SerializeToString(&raw);
  test_socket->connect_server();
  test_socket->send(raw);

  char buffer[1024] = {0};
  int num_read = read(test_pipe[0], buffer, 1024);
  std::string received_raw;
  received_raw.reserve(num_read);
  for (int i = 0; i < num_read; i++) {
    received_raw.push_back(buffer[i]);
  }
  EXPECT_EQ(raw, received_raw);
}

TEST_F(TrajectorySocketTest, receiveTest) {
  delete test_socket;
  test_socket = new TrajectorySocketStub(test_pipe[0]);
  std::string message = "OK";
  test_socket->connect_server();
  write(test_pipe[1], message.c_str(), message.size());
  std::string received = test_socket->receive();
  EXPECT_EQ(message, received);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
