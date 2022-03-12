#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>

#include "mock_class.h"
using ::testing::AtLeast;

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
    black = new player("name=mcts " + player_arg + " role=black");
    white = new player("name=mcts " + player_arg + " role=white");
    engine = new SelfPlayEngine(black, white);
  }

  void TearDown() override {
    delete engine;
    delete black;
    delete white;
  }

  SelfPlayEngine* engine;
  player* black;
  player* white;
};

TEST_F(SelfPlayEngineTest, OpenEpisodeTest) {
  // Test the episode.open_episode() has been called
  EpisodeMock episode_mock;
  std::string episode_str = black->name() + ":" + white->name();
  EXPECT_CALL(episode_mock, open_episode(episode_str))
    .Times(1);
  engine->init_game(&episode_mock);
  engine->open_episode();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
