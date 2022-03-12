#include <string>

#include "episode.h"
#include "action.h"
#include "agent.h"
#include "../proto/trajectory.pb.h"

class EngineInterface {
public:
    virtual ~EngineInterface() {}
    virtual void init_game(EpisodeInterface* game) = 0;
    virtual void open_episode() = 0;
    virtual void close_episode(std::string) = 0;
    virtual action next_action() = 0;
    virtual void store_transition(const action&) = 0;
};

class SelfPlayEngine : EngineInterface{
public:
    SelfPlayEngine(agent* black, agent* white) {
        
    }
    ~SelfPlayEngine() = default;
    void init_game(EpisodeInterface* game) override {
        game_ = game;
    }
    void open_episode() override {}
    void close_episode(std::string) override {}
    action next_action() override {}
    void store_transition(const action&) override {}

private:
    EpisodeInterface* game_;
    agent* black_;
    agent* white_;
};