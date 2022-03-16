#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>

#include "mock_class.h"
#include "../proto/trajectory.pb.h"
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

TEST_F(SelfPlayEngineTest, doNextActionTest) {
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
  action choosen_action = engine->do_next_action();
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

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
