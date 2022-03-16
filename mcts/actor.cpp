#include <iostream>
#include <string>

#include "episode.h"
#include "agent.h"
#include "proto/trajectory.pb.h"
#include "self_play.h"

// void self_play_loop(player& black, player& white) {
//     while (true) {
//         episode game;
//         game.open_episode(black.name() + ":" + white.name());
//         while (true) {
//             agent& who = game.take_turns(black, white);
//             action move = who.take_action(game.state());
//             if (game.apply_action(move) != true) break;
//             if (who.check_for_win(game.state())) break;
//         }
//         agent& win = game.last_turns(black, white);
//         game.close_episode(win.role());
//         std::cout << win.role() << std::endl;
//     }
// }

void self_play_loop(player& black, player& white) {
    EngineInterface* engine = new SelfPlayEngine(&black, &white);
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
        // TODO: send the raw trajectory to learner
        std::cout << winner.role() << std::endl;
    }
}

int main(int argc, const char* argv[]) {
    // SelfPlayEngine engine;
    std::cout << "=====Traning Self Play=====" << std::endl;
    std::string player_arg = 
        "simulation=1000 explore=0.3 uct=normal parallel=1";
    player black("name=mcts " + player_arg + " role=black");
    player white("name=mcts " + player_arg + " role=white");
    self_play_loop(black, white);
}