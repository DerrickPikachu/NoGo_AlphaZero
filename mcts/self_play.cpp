#include <iostream>
#include <string>

#include "episode.h"
#include "agent.h"
#include "trajectory.pb.h"

void self_play_loop(player& black, player& white) {
    while (true) {
        episode game;
        game.open_episode(black.name() + ":" + white.name());
        while (true) {
            agent& who = game.take_turns(black, white);
            action move = who.take_action(game.state());
            if (game.apply_action(move) != true) break;
            if (who.check_for_win(game.state())) break;
        }
        agent& win = game.last_turns(black, white);
        game.close_episode(win.role());
        std::cout << win.role() << std::endl;
    }
}

int main(int argc, const char* argv[]) {
    trajectory test;
    std::cout << "=====Traning Self Play=====" << std::endl;
    std::string player_arg = 
        "simulation=1000 explore=0.3 uct=normal parallel=1";
    player black("name=mcts " + player_arg + " role=black");
    player white("name=mcts " + player_arg + " role=white");
    self_play_loop(black, white);
}