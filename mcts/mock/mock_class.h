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

class EngineMock : public EngineInterface {
public:
  MOCK_METHOD1(init_game, void(EpisodeInterface* game));
  MOCK_METHOD0(open_episode, void());
  MOCK_METHOD1(close_episode, void(std::string));
  MOCK_METHOD0(next_action, action());
  MOCK_METHOD1(store_transition, void(const action&));
};