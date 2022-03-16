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
    virtual agent& close_episode() = 0;
    virtual action next_action() = 0;
    virtual bool apply_action(action move) = 0;
    virtual void store_transition(const action::place&) = 0;
};

class SelfPlayEngine : public EngineInterface{
public:
    SelfPlayEngine(agent* black, agent* white) {
        black_ = black;
        white_ = white;
    }
    ~SelfPlayEngine() = default;

    void init_game(EpisodeInterface* game) override {
        game_ = game;
    }
    
    void open_episode() override {
        std::string episode_msg = black_->name() + ":" + white_->name();
        game_->open_episode(episode_msg);
        trajectory_buffer.clear_transitions();
        add_new_trasition(game_->state());
    }

    agent& close_episode() override {
        agent& winner = game_->last_turns(*black_, *white_);
        game_->close_episode(winner.role());
        int num_transition = trajectory_buffer.transitions_size();
        auto* transition = trajectory_buffer.mutable_transitions(num_transition - 1);
        if (winner.role() == "black") {
            transition->set_reward(1.0);
        } else if (winner.role() == "white") {
            transition->set_reward(-1.0);
        }
        return winner;
    }

    action next_action() override {
        agent& who = game_->take_turns(*black_, *white_);
        action choosen_action = who.take_action(game_->state());
        return choosen_action;
    }

    bool apply_action(action move) override {
        return game_->apply_action(move);
    }
    
    void store_transition(const action::place& move) override {
        int num_trajectory = trajectory_buffer.transitions_size();
        trajectory::transition* last_transition = 
            trajectory_buffer.mutable_transitions(num_trajectory - 1);
        board::point point = move.position();
        last_transition->set_action_id(point.x * board::size_y + point.y);
        last_transition->set_reward(0.0);
        add_new_trasition(game_->state());
    }

    std::string get_trajectory() {
        std::string raw;
        trajectory_buffer.SerializeToString(&raw);
        return raw;
    }

private:
    void add_new_trasition(board& board_state) {
        trajectory::transition* first_transition = 
            trajectory_buffer.add_transitions();
        auto* state_proto = first_transition->mutable_state();
        for (int x = 0; x < board::size_x; x++) {
            for (int y = 0; y < board::size_y; y++) {
                float value;
                if (board_state[x][y] == 2u) {
                    value = -1.0;
                } else if (board_state[x][y] == 3u) {
                    value = 0.0;
                } else if (board_state[x][y] == -1u) {
                    // TODO: Throw exception
                    std::cerr << "error with unknow piece type on the board" << std::endl;
                    exit(0);
                } else {
                    value = board_state[x][y];
                }
                state_proto->Add(value);
            }
        }
    }

private:
    EpisodeInterface* game_;
    trajectory trajectory_buffer;
    agent* black_;
    agent* white_;
};