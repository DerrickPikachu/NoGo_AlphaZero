#include <gmock/gmock.h>

#include "../game/episode.h"
#include "../game/self_play.h"
#include "../game/agent.h"
#include "../game/board.h"
// #include "../game/alphazero_mcts.h"

class EpisodeMock : public EpisodeInterface {
public:
  MOCK_METHOD0(state, board&());
  MOCK_CONST_METHOD0(state, const board&());
  MOCK_CONST_METHOD0(score, board::reward());
  MOCK_METHOD1(open_episode, void(const std::string& tag));
  MOCK_METHOD1(close_episode, void(const std::string& tag));
  MOCK_METHOD1(apply_action, bool(action move));
  MOCK_METHOD2(take_turns, agent&(agent& black, agent& white));
  MOCK_METHOD2(last_turns, agent&(agent& black, agent& white));
  MOCK_CONST_METHOD1(step, size_t(unsigned who));
  MOCK_CONST_METHOD1(time, time_t(unsigned who));
  MOCK_CONST_METHOD1(actions, std::vector<action>(unsigned who));
};

class AgentMock : public agent {
public:
  MOCK_METHOD1(open_episode, void(const std::string& flag));
  MOCK_METHOD1(close_episode, void(const std::string& flag));
  MOCK_METHOD1(take_action, action(const board& b));
  MOCK_METHOD1(check_for_win, bool(const board& b));
  MOCK_CONST_METHOD1(property, std::string(const std::string& key));
  MOCK_METHOD1(notify, void(const std::string& msg));
  MOCK_CONST_METHOD0(name, std::string());
  MOCK_CONST_METHOD0(role, std::string());
};

class EngineMock : public EngineInterface {
public:
  MOCK_METHOD1(init_game, void(EpisodeInterface* game));
  MOCK_METHOD0(open_episode, void());
  MOCK_METHOD0(close_episode, agent&());
  MOCK_METHOD0(do_next_action, action());
  MOCK_METHOD1(apply_action, bool(action move));
  MOCK_METHOD1(store_transition, void(const action::place& move));
};

class PipeMock : public PipeInterface {
public:
  MOCK_METHOD1(write_to_pipe, bool(std::string data));
  MOCK_METHOD0(read_from_pipe, std::string());
  MOCK_METHOD0(close_write, void());
  MOCK_METHOD0(close_read, void());
  MOCK_METHOD0(redirect_stdout, void());
  MOCK_METHOD0(redirect_stdin, void());
};

class NetMock : public NetInterface {
public:
  MOCK_METHOD0(exec_net, void());
  MOCK_METHOD1(get_forward_result, std::string(const board&));
  MOCK_METHOD1(refresh_model, void(std::string));
  MOCK_METHOD0(send_exit, void());
  MOCK_METHOD1(parse_result, std::pair<std::vector<float>, float>(std::string));
};

class NodeMock :public NodeInterface {
public:
  MOCK_METHOD0(select, NodeInterface*());
  MOCK_METHOD1(expand, float(NetInterface*));
  MOCK_METHOD1(update, void(float));
  MOCK_METHOD0(value, float());
  MOCK_METHOD0(reset, void());
  MOCK_METHOD1(best_action, board::point(std::string));
  MOCK_METHOD0(get_state, board());
  MOCK_METHOD0(get_color, board::piece_type());
  MOCK_METHOD0(expanded, bool());
};
