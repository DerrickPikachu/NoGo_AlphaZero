#include <iostream>
#include <string>

#include "episode.h"
// #include "alphazero_mcts.h"
#include "agent.h"
#include "proto/trajectory.pb.h"
#include "self_play.h"
#include "trajectory_socket.h"


void self_play_loop(player& black, player& white) {
    EngineInterface* engine = new SelfPlayEngine(&black, &white);
    SocketInterface* learner_server = new TrajectorySocket("127.0.0.1", 7000);
    if (learner_server->connect_server()) {
        std::cerr << "connection succeed" << std::endl;
    } else {
        std::cerr << "connection failed" << std::endl;
        exit(0);
    }
    while (true) {
        engine->init_game(new episode());
        engine->open_episode();
        while(true) {
            action move = engine->next_action();
            if (engine->apply_action(move) == false) break;
            engine->store_transition(move);
        }
        agent& winner = engine->close_episode();
        std::string raw_trajectory = 
            ((SelfPlayEngine*)engine)->get_trajectory();
        trajectory tra;
        tra.ParseFromString(raw_trajectory);
        std::cerr << "num_transition: " << tra.transitions_size() << std::endl;
        int size = raw_trajectory.size();
        std::string byte_int(4, 0);
        for (int i = 3; i >= 0; i--) {
            byte_int[i] = size % (1 << 8);
            size = size >> 8;
        }
        learner_server->send(byte_int);
        learner_server->send(raw_trajectory);
        std::cerr << "send trajectory" << std::endl;
    }
}

int main(int argc, const char* argv[]) {
    std::cerr << "=====Traning Self Play=====" << std::endl;
    // std::string player_arg = 
    //     "simulation=1000 explore=0.3 uct=normal parallel=1";
    // player black("name=mcts method=mcts " + player_arg + " role=black");
    // player white("name=mcts method=mcts " + player_arg + " role=white");
    std::string player_arg = 
        "method=alphazero model=/desktop/mcts/game/model_provider/test_model mode=training simulation=200";
    player black("name=alphablack " + player_arg + " role=black");
    player white("name=alphawhite " + player_arg + " role=white");
    black.update_model("fake_weight.pth");
    white.update_model("fake_weight.pth");
    self_play_loop(black, white);
}