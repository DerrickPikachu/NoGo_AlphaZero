#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <exception>
#include <tuple>

#include "mock_class.h"
#include "../proto/trajectory.pb.h"
#include "trajectory_socket.h"
// #include "alphazero_mcts.h"
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

TEST_F(SelfPlayEngineTest, ApplyLegalActionTest) {
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

TEST_F(SelfPlayEngineTest, ApplyIllegalActionTest) {
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

TEST_F(SelfPlayEngineTest, StoreTransitionTest) {
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

TEST_F(SelfPlayEngineTest, FullTrajectoryTest) {
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
  // redundant
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

TEST_F(TrajectorySocketTest, SendTest) {
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

TEST_F(TrajectorySocketTest, ReceiveTest) {
  delete test_socket;
  test_socket = new TrajectorySocketStub(test_pipe[0]);
  std::string message = "OK";
  test_socket->connect_server();
  write(test_pipe[1], message.c_str(), message.size());
  std::string received = test_socket->receive();
  EXPECT_EQ(message, received);
}

class WrapNet : public AlphaZeroNet {
public:
  WrapNet(
    std::string net_path, std::string weight_path, int board_size) :
    AlphaZeroNet(net_path, weight_path, board_size) {}
  
  void set_write_pipe(PipeInterface* pipe) {
    write_pipe = pipe;
  }

  void set_read_pipe (PipeInterface* pipe) {
    read_pipe = pipe;
  }
};

class AlphaZeroNetTest : public ::testing::Test {
protected:
  void SetUp() override {
    alphazero_net = new WrapNet(
      "/desktop/mcts/game/model_provider/alphazero_net.py",
      "/desktop/mcts/game/model_provider/test_model/",
      9
    );
    write_pipe = new PipeMock();
    read_pipe = new PipeMock();
    alphazero_net->set_write_pipe(write_pipe);
    alphazero_net->set_read_pipe(read_pipe);
  }

  void TearDown() override {
    delete alphazero_net;
    if (write_pipe != NULL)
      delete write_pipe;
    if (read_pipe != NULL)
      delete read_pipe;
  }

public:
  WrapNet* alphazero_net;
  PipeMock* write_pipe;
  PipeMock* read_pipe;
};

TEST_F(AlphaZeroNetTest, ForwardResultTest) {
  board state;
  std::string fake_state;
  std::string fake_return;
  
  for (int i = 0; i < 81; i++) {
    fake_state += "0.0,";
  }
  fake_state.pop_back();
  
  for (int i = 0; i < 81; i++) {
    fake_return += "0.01234";
    if (i == 80)
      fake_return += ";";
    else
      fake_return += ",";
  }
  fake_return += "0.5";

  EXPECT_CALL(*write_pipe, write_to_pipe("forward\n"));
  EXPECT_CALL(*write_pipe, write_to_pipe(fake_state + "\n"));
  EXPECT_CALL(*read_pipe, read_from_pipe())
    .WillOnce(Return(fake_return));
  std::string result = alphazero_net->get_forward_result(state);
}

TEST_F(AlphaZeroNetTest, RefreshModelTest) {
  EXPECT_CALL(*write_pipe, write_to_pipe("refresh\n"));
  EXPECT_CALL(*write_pipe, write_to_pipe("test.pth\n"));
  alphazero_net->refresh_model("test.pth");
}

TEST_F(AlphaZeroNetTest, SendExitTest) {
  EXPECT_CALL(*write_pipe, write_to_pipe("exit\n"));
  EXPECT_CALL(*write_pipe, close_write());
  EXPECT_CALL(*read_pipe, close_read());
  alphazero_net->send_exit();
  write_pipe = NULL;
  read_pipe = NULL;
}

TEST_F(AlphaZeroNetTest, ParseResultTest) {
  std::string fake_result;
  for (int i = 0; i < 81; i++) {
    fake_result += "0.01234";
    if (i == 80)
      fake_result += ";";
    else
      fake_result += ",";
  }
  fake_result += "0.5";
  auto parsed_result = alphazero_net->parse_result(fake_result);
  for (auto& prob : parsed_result.first) {
    EXPECT_FLOAT_EQ(prob, 0.01234);
  }
  EXPECT_FLOAT_EQ(parsed_result.second, 0.5);
}

TEST_F(AlphaZeroNetTest, ExecNetTest) {
  alphazero_net->exec_net();
  alphazero_net->send_exit();
}

class PipeTest : public ::testing::Test {
public:
  PipeTest() {
    stdout_preserve = dup(STDOUT_FILENO);
    stdin_preserve = dup(STDIN_FILENO);
  }

protected:
  void SetUp() override {
    pipe = new Pipe();
  }

  void TearDown() override {
    delete pipe;
    dup2(stdout_preserve, STDOUT_FILENO);
    dup2(stdin_preserve, STDIN_FILENO);
  }

  PipeInterface* pipe;
  int stdout_preserve;
  int stdin_preserve;
};

TEST_F(PipeTest, ReadWriteTest) {
  std::string test_message = "read write test#";
  pipe->write_to_pipe(test_message);
  std::string result = pipe->read_from_pipe();
  test_message.pop_back();
  EXPECT_EQ(result, test_message);
}

TEST_F(PipeTest, CloseWriteTest) {
  pipe->close_write();
  EXPECT_THROW({
    try {
      pipe->write_to_pipe("test");
    } catch (const std::exception& e) {
      EXPECT_STREQ("Write after close write", e.what());
      throw;
    }
  }, std::exception);
}

TEST_F(PipeTest, CloseReadTest) {
  pipe->close_read();
  EXPECT_THROW({
    try {
      std::string result = pipe->read_from_pipe();
    } catch (const std::exception &e) {
      EXPECT_STREQ("Read after close read", e.what());
      throw;
    }
  }, std::exception);
}

TEST_F(PipeTest, RedirectStdoutTest) {
  std::string test_message = "test redirect stdout#";
  pipe->redirect_stdout();
  std::cout << test_message << std::flush;
  std::string result = pipe->read_from_pipe();
  test_message.pop_back();
  EXPECT_EQ(result, test_message);
}

TEST_F(PipeTest, RedirectStdinTest) {
  std::string test_message = "test redirect stdin\n";
  pipe->redirect_stdin();
  pipe->write_to_pipe(test_message);
  std::string result;
  std::getline(std::cin, result);
  test_message.pop_back();
  EXPECT_EQ(result, test_message);
}

class NodeTest : public ::testing::Test {
public:
  class WrapNode : public Node {
  public:
    WrapNode(const board& b, board::piece_type color) : 
      Node(b, color) {}
    
    std::vector<std::tuple<float, board::point, Node>>* get_childs() {
      return &childs;
    }

    void set_is_expand(bool value) { is_expand = value; }
  };

protected:
  void SetUp() override {
    board fake_board;
    test_node = new WrapNode(fake_board, board::piece_type::black);
    node_childs = test_node->get_childs();
  }

  void TearDown() override {
    delete test_node;
  }

public:
  WrapNode* test_node;
  std::vector<std::tuple<float, board::point, Node>>* node_childs;
};

TEST_F(NodeTest, SelectTestWithInputAllZero) {
  test_node->update(0);
  board ans_board = test_node->get_state();
  ans_board.place(board::point(0));
  for (int i = 0; i < 5; i++) {
    board tem = test_node->get_state();
    board::point action = board::point(i);
    tem.place(action);
    node_childs->push_back({
      1.0 / 81,
      action,
      Node(tem, board::piece_type::white)
    });
  }
  test_node->set_is_expand(true);
  NodeInterface* selected_node = test_node->select();
  if (selected_node != NULL)
    std::cout << selected_node->get_state() << std::endl;
  EXPECT_TRUE(ans_board == selected_node->get_state());
}

TEST_F(NodeTest, SelectTestWithInputDiff) {
  board ans_board = test_node->get_state();
  ans_board.place(board::point(2));
  std::vector<float> action_prob = { 0.2, 0.1, 0.4, 0.3 };
  for (int i = 0; i < 4; i++) {
    board tem = test_node->get_state();
    board::point action(i);
    tem.place(action);
    node_childs->push_back({
      action_prob[i],
      action,
      Node(tem, board::piece_type::white)
    });
  }
  test_node->update(0.0);  // make visit count = 1
  test_node->set_is_expand(true);
  NodeInterface* selected_node = test_node->select();
  EXPECT_TRUE(ans_board == selected_node->get_state());
}

TEST_F(NodeTest, ExpandTest) {
  std::vector<float> fake_policy;
  int total_sum = (1 + 81) * 81 / 2;
  for (int i = 0; i < 81; i++) {
    fake_policy.push_back((i + 1) / (float)total_sum);
  }
  float fake_value = 0.6;
  std::pair<std::vector<float>, float> fake_return(
    fake_policy, fake_value
  );
  NetMock net_mock;
  std::string fake_result = "test";
  EXPECT_CALL(net_mock, get_forward_result(test_node->get_state()))
    .WillOnce(Return(fake_result));
  EXPECT_CALL(net_mock, parse_result(fake_result))
    .WillOnce(Return(fake_return));
  
  // Test target
  float winrate = test_node->expand(&net_mock);

  EXPECT_FLOAT_EQ(winrate, fake_value);
  for (auto& child : *node_childs) {
    float prob = std::get<0>(child);
    board::point action = std::get<1>(child);
    Node& node = std::get<2>(child);
    board tem = test_node->get_state();
    tem.place(action);
    EXPECT_FLOAT_EQ(fake_policy[action.i], prob);
    EXPECT_TRUE(tem == node.get_state());
    EXPECT_EQ(board::piece_type::white, node.get_color());
  }
  EXPECT_EQ(node_childs->size(), 81 - 9);

  // expand again test
  EXPECT_THROW({
    try {
      test_node->expand(NULL);
    } catch (const std::exception& e) {
      EXPECT_STREQ(
        "Expand error: The node has been expanded, but try to expand again",
        e.what());
      throw;
    }
  }, AlphaZeroException);
}

TEST_F(NodeTest, ExpandWithEndStateTest) {
  int action_space = board::size_x * board::size_y - 1;
  int i = 0;
  board test_board;
  while (true) {
    if (test_board(i) != 3u) {
      test_board.place(board::point(i));
      if (test_board.place(board::point(action_space - i)) != board::legal) break;
    }
    i++;
  }
  std::cout << test_board << std::endl;
  Node end_state_node(test_board, board::piece_type::white);
  NetMock net_mock;
  float value = end_state_node.expand(&net_mock);
  EXPECT_FLOAT_EQ(1.0, value);
}

TEST_F(NodeTest, UpdateTestOnlyCallOnce) {
  test_node->update(0.8);
  EXPECT_FLOAT_EQ(0.8, test_node->value());
}

TEST_F(NodeTest, UpdateTestCallMultipleTimes) {
  test_node->update(0.9);
  test_node->update(0.4);
  test_node->update(0.6);
  EXPECT_FLOAT_EQ((0.9 + 0.4 + 0.6) / 3, test_node->value());
  
}

TEST_F(NodeTest, BestActionTest) {
  std::vector<int> simulation_count = {
    19, 15, 55, 77, 3
  };
  for (int i = 0; i < 5; i++) {
    board tem = test_node->get_state();
    board::point action(i);
    tem.place(action);
    Node child(tem, board::piece_type::white);
    for (int j = 0; j < simulation_count[i]; j++) {
      child.update(0.0);  // accumulate visit count
    }
    node_childs->push_back({
      0.0,
      action,
      child
    });
  }
  std::default_random_engine generator;

  // Test target
  board::point ans = test_node->best_action("evaluating", generator);

  EXPECT_EQ(3, ans.i);
}

TEST_F(NodeTest, BestActionWithTrainingModeTest) {
  std::vector<int> simulation_count = {
    19, 15, 55, 77, 3
  };
  for (int i = 0; i < 5; i++) {
    board tem = test_node->get_state();
    board::point action(i);
    tem.place(action);
    Node child(tem, board::piece_type::white);
    for (int j = 0; j < simulation_count[i]; j++) {
      child.update(0.0);  // accumulate visit count
    }
    node_childs->push_back({
      0.0,
      action,
      child
    });
  }

  std::vector<int> action_counter(5);
  std::default_random_engine generator;
  for (int i = 0; i < 1000; i++) {
    board::point action = test_node->best_action("training", generator);
    action_counter[action.i]++;
  }
  std::vector<float> probs(5);
  float total = 0.0;
  for (int i = 0; i < 5; i++) {
    float tem = pow(simulation_count[i], 2);
    probs[i] = tem;
    total += tem;
  }
  for (int i = 0; i < 5; i++) {
    probs[i] /= total;
  }

  for (int i = 0; i < 5; i++) {
    std::cerr << action_counter[i] << " ";
    int expected = probs[i] * 1000;
    int miss = abs(expected - action_counter[i]);
    EXPECT_TRUE(miss < 50);
  }
  std::cerr << std::endl;
}

TEST_F(NodeTest, BestActionWhenNoChilds) {
  std::default_random_engine generator;
  board::point ans = test_node->best_action("evaluating", generator);
  EXPECT_EQ(-1, ans.i);
}

class WhiteNodeTest : public ::testing::Test {
protected:
  void SetUp() override {
    board fake_board;
    fake_board.place(board::point(0));
    test_node = new NodeTest::WrapNode(fake_board, board::piece_type::white);
    node_childs = test_node->get_childs();
  }
  
  void TearDown() override {
    delete test_node;
  }

public:
  NodeTest::WrapNode* test_node;
  std::vector<std::tuple<float, board::point, Node>>* node_childs;
};

TEST_F(WhiteNodeTest, SelectTestWithNegitiveWinrate) {
  board ans_board = test_node->get_state();
  ans_board.place(board::point(1));
  test_node->update(0.0);
  test_node->update(0.0);  // make visit count = 2
  std::vector<float> fake_value = { -0.6, -0.3, 0.1, 0.6, -0.4 };
  for (int i = 0; i < fake_value.size(); i++) {
    board tem = test_node->get_state();
    board::point action(i + 1);
    tem.place(action);
    Node child(tem, board::piece_type::black);
    child.update(fake_value[i]);
    node_childs->push_back({
      0.0,
      action,
      child
    });
  }
  test_node->set_is_expand(true);

  NodeInterface* selected_node = test_node->select();
  EXPECT_TRUE(ans_board == selected_node->get_state());
}

TEST_F(WhiteNodeTest, SelectTestWithNoChild) {
  EXPECT_THROW({
    try {
      test_node->select();
    } catch (const std::exception& e) {
      EXPECT_STREQ("Select error: node hasn't been expand", e.what());
      throw;
    }
  }, AlphaZeroException);
}

TEST_F(WhiteNodeTest, SelectTestWithEndState) {
  int action_space = board::size_x * board::size_y - 1;
  int i = 0;
  board test_board;
  while (true) {
    if (test_board(i) != 3u) {
      test_board.place(board::point(i));
      if (test_board.place(board::point(action_space - i)) != board::legal) break;
    }
    i++;
  }
  test_node->set_is_expand(true);
  NodeInterface* select_result = test_node->select();
  EXPECT_EQ(NULL, select_result);
}

TEST_F(WhiteNodeTest, ExpandTestWithWhiteRoot) {
  std::vector<float> fake_policy;
  int total_sum = (1 + 81) * 81 / 2;
  for (int i = 0; i < 81; i++) {
    fake_policy.push_back((i + 1) / (float)total_sum);
  }
  float fake_value = 0.6;
  std::pair<std::vector<float>, float> fake_return(
    fake_policy, fake_value
  );
  NetMock net_mock;
  std::string fake_result = "test";
  EXPECT_CALL(net_mock, get_forward_result(test_node->get_state()))
    .WillOnce(Return(fake_result));
  EXPECT_CALL(net_mock, parse_result(fake_result))
    .WillOnce(Return(fake_return));
  
  float winrate = test_node->expand(&net_mock);

  for (auto& child : *node_childs) {
    Node& node = std::get<2>(child);
    EXPECT_EQ(board::piece_type::black, node.get_color());
  }
  EXPECT_EQ(node_childs->size(), 81 - 10);
}


class TreeTest : public ::testing::Test {
public:
  class WrapTree : public Tree {
  public:
    WrapTree(NetInterface* network_provider, std::string mcts_mode) :
     Tree(network_provider, mcts_mode) {}

    void set_history(std::vector<NodeInterface*> fake_history) {
      history = fake_history;
    }

    std::vector<NodeInterface*> get_history() { return history; }
    NodeInterface* get_select_node() { return select_node; }
    void set_select_node(NodeInterface* node) { select_node = node; }
  };

  class WrapNode : public Node {
  public:
    WrapNode(board b, board::piece_type piece) :
      Node(b, piece) {}

    void set_childs(
      std::vector<std::tuple<float, board::point, WrapNode>>& fake_child) {
      childs = fake_child;
    }

    void set_is_expand(bool value) { is_expand = value; }

  protected:
    std::vector<std::tuple<float, board::point, WrapNode>> childs;
  };

protected:
  void SetUp() override {
    tree = new Tree(&fake_net, "evaluating");
  }

  void TearDown() override {
    delete tree;
  }

public:
  TreeInterface* tree;
  NodeInterface* test_root;
  NetMock fake_net;
};

TEST_F(TreeTest, SelectTestWithOnlyRoot) {
  test_root = new WrapNode(board(), board::piece_type::black);
  tree->set_root(test_root);
  EXPECT_THROW({
    try {
      tree->select();
    } catch (const std::exception& c) {
      EXPECT_STREQ(
        "Select error: root haven't been expanded",
        c.what());
      throw;
    }
  }, AlphaZeroException);
}

TEST_F(TreeTest, SelectTestWithOneLevel) {
  std::vector<NodeMock> fake_nodes(2);
  for (int i = 0; i < fake_nodes.size() - 1; i++) {
    EXPECT_CALL(fake_nodes[i], select())
      .WillOnce(Return(&fake_nodes[i+1]));
    EXPECT_CALL(fake_nodes[i], expanded())
      .WillRepeatedly(Return(true));
  }
  EXPECT_CALL(fake_nodes.back(), get_state())
    .WillOnce(Return(board()));
  EXPECT_CALL(fake_nodes.back(), expanded())
    .WillRepeatedly(Return(false));
  test_root = &fake_nodes[0];
  tree->set_root(test_root);
  tree->select();

  NodeInterface* select_node = ((WrapTree*)tree)->get_select_node();
  EXPECT_TRUE(board() == select_node->get_state());
}

TEST_F(TreeTest, SelectTestWithMultiLevel) {
  std::vector<NodeMock> fake_nodes(10);
  for (int i = 0; i < fake_nodes.size() - 1; i++) {
    EXPECT_CALL(fake_nodes[i], select())
      .WillOnce(Return(&fake_nodes[i+1]));
    EXPECT_CALL(fake_nodes[i], expanded())
      .WillRepeatedly(Return(true));
    board tem;
    tem.place(board::point(i));
    EXPECT_CALL(fake_nodes[i], get_state())
      .WillRepeatedly(Return(tem));
  }
  EXPECT_CALL(fake_nodes.back(), get_state())
    .WillRepeatedly(Return(board()));
  EXPECT_CALL(fake_nodes.back(), expanded())
    .WillRepeatedly(Return(false));
  test_root = &fake_nodes[0];
  tree->set_root(test_root);
  tree->select();

  NodeInterface* select_node = ((WrapTree*)tree)->get_select_node();
  EXPECT_TRUE(board() == select_node->get_state());
  auto result_history = ((WrapTree*)tree)->get_history();
  for (int i = 0; i < fake_nodes.size(); i++) {
    EXPECT_EQ(fake_nodes[i].get_state(), result_history[i]->get_state());
  }
}

TEST_F(TreeTest, SelectTestEncounterEndState) {
  std::vector<NodeMock> fake_nodes(10);
  for (int i = 0; i < fake_nodes.size() - 1; i++) {
    EXPECT_CALL(fake_nodes[i], select())
      .WillOnce(Return(&fake_nodes[i+1]));
    EXPECT_CALL(fake_nodes[i], expanded())
      .WillRepeatedly(Return(true));
  }
  EXPECT_CALL(fake_nodes.back(), select())
    .WillOnce(Return(nullptr));
  EXPECT_CALL(fake_nodes.back(), expanded())
    .WillOnce(Return(true));
  EXPECT_CALL(fake_nodes.back(), get_state())
    .WillOnce(Return(board()));
  test_root = &fake_nodes[0];
  tree->set_root(test_root);
  tree->select();

  NodeInterface* select_node = ((WrapTree*)tree)->get_select_node();
  EXPECT_TRUE(board() == select_node->get_state());
}

TEST_F(TreeTest, ExpandTest) {
  NodeMock mock_node;
  EXPECT_CALL(mock_node, expand(&fake_net))
    .WillOnce(Return(0.7));
  ((WrapTree*)tree)->set_select_node(&mock_node);
  float value = tree->expand();
  EXPECT_FLOAT_EQ(0.7, value);
}

TEST_F(TreeTest, ExpandTestWithNullNode) {
  EXPECT_THROW({
    try {
      tree->expand();
    } catch (std::exception& c) {
      EXPECT_STREQ(
        "Expand error: tree select node is null",
        c.what()
      );
      throw;
    }
  }, AlphaZeroException);
}

TEST_F(TreeTest, UpdateTestWithEmptyHistory) {
  EXPECT_THROW({
    try {
      tree->update(0.7);
    } catch (std::exception& c) {
      EXPECT_STREQ(
        "Update error: the history is empty",
        c.what()
      );
      throw;
    }
  }, AlphaZeroException);
}

TEST_F(TreeTest, UpdateOnlyOneNode) {
  float winrate = 0.3;
  std::vector<NodeInterface*> fake_history;
  NodeMock fake_node;
  EXPECT_CALL(fake_node, update(winrate));
  fake_history.push_back(&fake_node);
  ((WrapTree*)tree)->set_history(fake_history);
  tree->update(winrate);
}

TEST_F(TreeTest, UpdateMultiNode) {
  float winrate = 0.3;
  std::vector<NodeInterface*> fake_history;
  for (int i = 0; i < 10; i++) {
    NodeMock* fake_node = new NodeMock();
    EXPECT_CALL(*fake_node, update(winrate));
    fake_history.push_back(fake_node);
  }
  ((WrapTree*)tree)->set_history(fake_history);
  tree->update(winrate);
  for (int i = 0; i < 10; i++) {
    delete (NodeMock*)fake_history[i];
  }
  auto history = ((WrapTree*)tree)->get_history();
  EXPECT_TRUE(history.empty());
}

TEST_F(TreeTest, GetActionTest) {
  NodeMock fake_node;
  std::default_random_engine generator;
  EXPECT_CALL(fake_node, best_action("evaluating", generator))
    .WillOnce(Return(board::point(5)));
  tree->set_root(&fake_node);
  board::point move = tree->get_action(generator);
  EXPECT_TRUE(board::point(5).i == move.i);
}

class PlayerTest : public ::testing::Test {
public:
  class FakePlayer : public player {
  public:
    FakePlayer(const std::string& args="") : player(args) {}
    action randomAction(const board& state) override {
      return action::place(board::point(0), board::black);
    }
    action mctsAction(const board& state) override {
      return action::place(board::point(1), board::black);
    }
    action zeroAction(const board& state) override {
      return action::place(board::point(2), board::black);
    }
  };

protected:
  void SetUp() override {
    test_player = nullptr;
  }
  void TearDown() override {
    if (test_player != nullptr) {
      test_player->exit();
      delete test_player;
    }
  }

public:
  player* test_player;
};
// player args: name=agent_name method=zero model=model_path simulation=1000
// when ((method == zero || method == alphazero) && model != empty )
// then use alphazero agent
// otherwise use random agent or mcts agent
TEST_F(PlayerTest, PCTakeActionTest1) {
  // PC true test case
  test_player = new FakePlayer(
    "name=test_agent method=zero model=/abc simulation=1000 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(2).i);
}

TEST_F(PlayerTest, PCTakeActionTest2) {
  // PC false test case
  test_player = new FakePlayer(
    "name=test_agent method=zero simulation=1 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(0).i);
}

TEST_F(PlayerTest, CCTakeActionTest1) {
  // CC TFT test case
  test_player = new FakePlayer(
    "name=test_agent method=zero model=/abc simulation=1000 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(2).i);
}

TEST_F(PlayerTest, CCTakeActionTest2) {
  // CC FTF test case
  test_player = new FakePlayer(
    "name=test_agent method=alphazero simulation=11 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(0).i);
}

TEST_F(PlayerTest, CACCTakeActionTest1) {
  // CACC TFT test case
  test_player = new FakePlayer(
    "name=test_agent method=zero model=/abc simulation=1000 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(2).i);
}

TEST_F(PlayerTest, CACCTakeActionTest2) {
  // CACC FFT test case
  test_player = new FakePlayer(
    "name=test_agent method=mcts model=/abc simulation=1000 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(1).i);
}

TEST_F(PlayerTest, CACCTakeActionTest3) {
  // CACC FTF test case
  test_player = new FakePlayer(
    "name=test_agent method=alphazero simulation=11 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(0).i);
}

TEST_F(PlayerTest, CACCTakeActionTest4) {
  // CACC FTT test case
  test_player = new FakePlayer(
    "name=test_agent method=alphazero model=/abc simulation=11 role=black");
  action::place result = test_player->take_action(board());
  EXPECT_EQ(result.position().i, board::point(2).i);
}

/*
* Integration Test
* In this part of test, all test cases are designed to test the correctness
* of different class interaction.
*/
class NetToProviderTest : public ::testing::Test {
protected:
  void SetUp() override {
    net = new AlphaZeroNet(
      "/desktop/mcts/game/model_provider/alphazero_net.py",
      "/desktop/mcts/game/model_provider/test_model",
      9
    );
    net->exec_net();
  }

  void TearDown() override {
    net->send_exit();
    delete net;
  }

public:
  NetInterface* net;
};

TEST_F(NetToProviderTest, ForwardBoardTest) {
  board test_board;
  net->refresh_model("fake_weight.pth");
  std::string result = net->get_forward_result(test_board);
  // std::cout << "forward result: " << result << std::endl;
  auto policy_value = net->parse_result(result);
  std::vector<float> policy = policy_value.first;
  float value = policy_value.second;
  float policy_sum = 0;

  std::cerr << "policy: " << std::endl;
  for (int i = 0; i < policy.size(); i++) {
    printf("%.3f ", policy[i]);
    if (i % board::size_y == board::size_y - 1)
      std::cerr << std::endl;
  }
  std::cerr << "value: " << value << std::endl;

  for (int i = 0; i < policy.size(); i++)
    policy_sum += policy[i];
  EXPECT_FLOAT_EQ(1.0, policy_sum);
  EXPECT_TRUE(-1 <= value && value <= 1);
  EXPECT_EQ(81, policy.size());
}

TEST_F(NetToProviderTest, ModelRefreshTest) {
  board test_board;
  std::string pre_result = net->get_forward_result(test_board);
  net->refresh_model("fake_weight.pth");
  std::string result = net->get_forward_result(test_board);
  EXPECT_FALSE(result == pre_result);
}

TEST_F(NetToProviderTest, ForwardTwoBoardTest) {
  board test_board;
  net->refresh_model("fake_weight.pth");
  std::string result = net->get_forward_result(test_board);
  test_board.place(board::point(1));
  std::string result2 = net->get_forward_result(test_board);
  EXPECT_NE(result, result2);
}

class TreeSearchTest : public ::testing::Test {
protected:
  void SetUp() override {
    net = new AlphaZeroNet(
      "/desktop/mcts/game/model_provider/alphazero_net.py",
      "/desktop/mcts/game/model_provider/test_model",
      9
    );
    net->exec_net();
    net->refresh_model("fake_weight.pth");
    tree = new Tree(net, "evaluating");
  }
  void TearDown() override {
    net->send_exit();
    delete tree;
    delete net;
  }

public:
  NetInterface* net;
  TreeInterface* tree;
  std::default_random_engine engine;
};

TEST_F(TreeSearchTest, MCTSTest) {
  Node* node = new Node(board(), board::piece_type::black);
  float root_value = node->expand(net);
  node->update(root_value);
  tree->set_root(node);
  for (int i = 0; i < 500; i++) {
    tree->select();
    float value = tree->expand();
    tree->update(value);
  }
  auto childs = ((NodeTest::WrapNode*)node)->get_childs();
  for (int i = 0; i < childs->size(); i++) {
    Node child = std::get<2>(childs->at(i));
    std::cout << child.get_visit_count() << " ";
  }
  std::cout << std::endl;
  delete node;
}

TEST_F(TreeSearchTest, MCTSWithLargeSimulationTest) {
  Node* node = new Node(board(), board::piece_type::black);
  float root_value = node->expand(net);
  node->update(root_value);
  tree->set_root(node);
  for (int i = 0; i < 5000; i++) {
    tree->select();
    float value = tree->expand();
    tree->update(value);
  }
  auto childs = ((NodeTest::WrapNode*)node)->get_childs();
  for (int i = 0; i < childs->size(); i++) {
    Node child = std::get<2>(childs->at(i));
    std::cout << child.get_visit_count() << " ";
  }
  std::cout << std::endl;
  delete node;
}

TEST_F(TreeSearchTest, MCTSCheckMemoryLeak) {
  board test_board;
  for (int i = 0; i < 10; i++) {
    Node* node = new Node(test_board, board::piece_type::black);
    float root_value = node->expand(net);
    tree->set_root(node);
    for (int i = 0; i < 1000; i++) {
      tree->select();
      float value = tree->expand();
      tree->update(value);
    }

    board::point move = tree->get_action(engine);
    test_board.place(move);
    tree->reset();
  }
}

class ModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    net = new AlphaZeroNet(
      "/desktop/mcts/game/model_provider/alphazero_net.py",
      "/desktop/mcts/game/model_provider/test_model",
      9
    );
    net->exec_net();
    net->refresh_model("fake_weight.pth");
    black = new player("name=random method=random role=black seed=553");
    white = new player("name=random method=random role=white seed=23534");
  }

  void TearDown() override {
    net->send_exit();
    delete net;
    delete black;
    delete white;
    test_board = board();
  }

public:
  NetInterface* net;
  player* white;
  player* black;
  board test_board;
};

TEST_F(ModelTest, EndStateTest) {
  int step = 0;
  while (true) {
    action move = (step % 2 == 0)? 
      black->take_action(test_board) :
      white->take_action(test_board);
    if (move.apply(test_board) != board::legal)
      break;
    step++;
  }
  float win_value = (step % 2 == 0)? -1 : 1;
  std::cerr << "win_value: " << win_value << std::endl;
  std::cerr << test_board << std::endl;
  std::string result = net->get_forward_result(test_board);
  auto policy_value = net->parse_result(result);
  std::vector<float> policy = policy_value.first;
  float value = policy_value.second;

  std::cerr << "policy: " << std::endl;
  for (int i = 0; i < policy.size(); i++) {
    printf("%.3f ", policy[i]);
    if (i % board::size_y == board::size_y - 1)
      std::cerr << std::endl;
  }
  std::cerr << "value: " << value << std::endl;
}

TEST_F(ModelTest, NormalBoardTest) {
  int step = 0;
  while (true) {
    action move = (step % 2 == 0)? 
      black->take_action(test_board) :
      white->take_action(test_board);
    move.apply(test_board);
    step++;
    if (step == 40)
      break;
  }
  std::cerr << test_board << std::endl;
  std::string result = net->get_forward_result(test_board);
  auto policy_value = net->parse_result(result);
  std::vector<float> policy = policy_value.first;
  float value = policy_value.second;

  std::cerr << "policy: " << std::endl;
  for (int i = 0; i < policy.size(); i++) {
    printf("%.3f ", policy[i]);
    if (i % board::size_y == board::size_y - 1)
      std::cerr << std::endl;
  }
  std::cerr << "value: " << value << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
