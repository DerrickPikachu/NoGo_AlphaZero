#include <gmock/gmock.h>

#include "../game/episode.h"
#include "../game/self_play.h"
#include "../game/agent.h"
#include "../game/board.h"

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