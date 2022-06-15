/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <thread>
#include <future>
#include <ctime>

#include "board.h"
#include "action.h"
#include "mcts.h"
#include "alphazero_mcts.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

	bool operator ==(const agent& other) const {
		return this->name() == other.name() && this->role() == other.role();
	}

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : 
		random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty),
		net(nullptr), mcts_tree(nullptr) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		if (meta.count("model")) {
			std::string path = std::string(meta["model"]);
			// TODO: maybe change the net script path
			net = new AlphaZeroNet(
				"/desktop/mcts/game/model_provider/alphazero_net.py",
				path,
				board::size_x
			);
			net->exec_net();
			std::string mode = std::string(meta["mode"]);
			mcts_tree = new Tree(net, mode);
			if (mode == "evaluating") {
				update_model(std::string(meta["model_name"]));
			}
		}
	}

	virtual action take_action(const board& state) {
		std::string method = std::string(meta["method"]);
		if ((method == "zero" || method == "alphazero") && meta.count("model"))
			return zeroAction(state);
		else if (method == "mcts") {
		    return mctsAction(state);
		} else {
		    return randomAction(state);
		}
	}

	virtual action randomAction(const board& state) {
        std::shuffle(space.begin(), space.end(), engine);
        for (const action::place& move : space) {
            board after = state;
            if (move.apply(after) == board::legal)
                return move;
        }
        return action();
	}

	virtual action mctsAction(const board& state) {
        int parallel = int(meta["parallel"]);
        int actionSize = board::size_x * board::size_y;
        std::vector<Mcts> mcts(parallel);
        std::vector<std::thread> threads;
        for (int i = 0; i < parallel; i++) {
            threads.push_back(std::thread(&player::runMcts, this, state, &mcts[i]));
        }
        for (int i = 0; i < parallel; i++) {
            threads[i].join();
        }
        int bestCount = 0;
        int bestMoveIndex = 0;
        for (int i = 0; i < actionSize; i++) {
            int total = 0;
            for (int j = 0; j < parallel; j++)
                total += mcts[j].getSimulationCount(i);
            if (bestCount < total) {
                bestCount = total;
                bestMoveIndex = i;
            }
        }
        return action::place(board::point(bestMoveIndex), who);
	}

	virtual action zeroAction(const board& state) {
		Node* root = new Node(state, who);
		root->expand(net);
		if (root->is_end_state())
			return action::place(board::point(0), who);
		mcts_tree->set_root(root);
		for (int i = 0; i < int(meta["simulation"]); i++) {
			mcts_tree->select();
			float value = mcts_tree->expand();
			mcts_tree->update(value);
		}
		board::point move = mcts_tree->get_action(engine);
		std::cerr << "[selected move: " << move.i << "]" << std::endl;
		mcts_tree->reset();
		return action::place(move, who);
	}

    void runMcts(board state, Mcts* mcts) {
        mcts->setWho(who);
        mcts->setUctType(meta["uct"]);
//        mcts->setupRoot(state);
        mcts->search(state, int(meta["simulation"]), float(meta["explore"]));
    }

	void update_model(std::string model_name) {
		net->refresh_model(model_name);
	}

	void exit() {
		if (net != nullptr) {
			net->send_exit();
			delete net;
		}
		delete mcts_tree;
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
	Tree* mcts_tree;
	NetInterface* net;
};
