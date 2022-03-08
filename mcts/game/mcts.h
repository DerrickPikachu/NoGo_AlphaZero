//
// Created by pikachin on 2021/12/6.
//

#ifndef TCG_NOGO_MCTS_MCTS_H
#define TCG_NOGO_MCTS_MCTS_H

#include <unordered_map>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include <random>
#include <ctime>

// name=mcts simulation=10000 explore=0.5 uct=normal parallel=4

class Mcts {
private:
    struct Node {
        int visitCount;
        int wins;
        int raveCount;
        int raveWins;
        board position;
        board::point fromWhichMove;
        std::vector<Node*> mapActionToChild;
        std::vector<board::point> legal;
        std::vector<Node*> childs;
        Node(board b, std::default_random_engine& engine) : visitCount(0), wins(0), raveCount(0), raveWins(0), position(b) {
            mapActionToChild.resize(board::size_x * board::size_y, NULL);
            for (int i = 0; i < board::size_x * board::size_y; i++) {
                board::point move(i);
                board tem = b;
                if (tem.place(move) == board::legal)
                    legal.push_back(move);
            }
            std::shuffle(legal.begin(), legal.end(), engine);
        }
    };

public:
    Mcts() : uniform(0, board::size_x * board::size_y) {
        srand(time(NULL));
        engine.seed(rand() % 100000);
        int actionSize = board::size_x * board::size_y;
        traverseHistory.resize(actionSize, 0);
        actions.reserve(actionSize);
        for (int i = 0; i < actionSize; i++)
            actions.push_back(board::point(i));
        // TODO: Need test
    }
    Mcts(board::piece_type type) : Mcts() {
        setWho(type);
    }

    void setWho(board::piece_type type) {
        who = type;
    }

    void setUctType(std::string type) {
        uctType = type;
    }

    void setupRoot(const board& b) {
        nodePool.push_back(Node(b, engine));
        root = &nodePool.back();
        int result = simulate(root->position, false);
        update(root, result);
    }

    void resetMcts(Node* node=nullptr) {
        if (node == nullptr)
            node = root;
        for (int i = 0; i < (int)node->childs.size(); i++)
            resetMcts(node->childs[i]);
        delete node;
    }

    void search(const board& b, int timesOfMcts, float constant) {
        nodePool.reserve(timesOfMcts + 2);
        nodePool.push_back(Node(b, engine));
        root = &nodePool.back();
        exploreC = constant;
        for (int i = 0; i < timesOfMcts; i++) {
            traverse(root);
            cleanHistory();
        }
    }

    board::point chooseAction() {
        if (root->childs.empty())
            return board::point(0);
        int bestCount = 0;
        Node* bestNode = root->childs[0];
        for (int i = 0; i < (int)root->childs.size(); i++) {
            if (bestCount < root->childs[i]->visitCount) {
                bestCount = root->childs[i]->visitCount;
                bestNode = root->childs[i];
            }
        }
        return findActionByNextBoard(bestNode->position);
    }

    int getSimulationCount(int actionIndex) {
        if (root->mapActionToChild[actionIndex] == NULL)
            return 0;
        else
            return root->mapActionToChild[actionIndex]->visitCount;
    }

private:  // After testing, it should be private
    int traverse(Node* node, bool isOpponent=false) {
        if (!node->legal.empty()) {  // expand and simulate
            Node* leaf = expand(node, isOpponent);
            int result = simulate(leaf->position, !isOpponent);
            update(leaf, result);
            update(node, result);
            return result;
        } else {
            int result;
            if (node->childs.empty()) {  // Terminal node
                result = simulate(node->position, isOpponent);
            } else {
                Node* nextNode = select(node, isOpponent);
                result = traverse(nextNode, !isOpponent);
                traverseHistory[nextNode->fromWhichMove.i] = 1;
            }
//            std::cout << nextNode->position << std::endl;
            update(node, result);
            return result;
        }
    }

    Node* select(Node* node, bool isOpponent) {
        float bestScore = 0;
        std::vector<Node*> nextNodes;
        for (Node* child : node->childs) {
            float score = uct(*child, node->visitCount, isOpponent);
            if (bestScore < score) {
                bestScore = score;
                nextNodes.clear();
                nextNodes.push_back(child);
            } else if (bestScore == score) {
                nextNodes.push_back(child);
            }
        }
//        std::shuffle(nextNodes.begin(), nextNodes.end(), engine);
        if (nextNodes.empty()) {
            std::cerr << "select error" << std::endl;
            exit(0);
        }
        return nextNodes[0];
    }

    int simulate(const board& position, bool isOpponent) {
        std::vector<board::point> emptyPoint;
        int amountOfActions = board::size_x * board::size_y;
        for (int i = 0; i < amountOfActions; i++) {
            board::point move(i);
            if (position[move.x][move.y] == board::empty)
                emptyPoint.push_back(move);
        }
        board curPosition = position;
        int n = emptyPoint.size();
        board::point randomMove = getRandomAction(curPosition, emptyPoint, n);
        while (curPosition.place(randomMove) == board::legal) {
//            std::cout << curPosition << std::endl;
            n--;
            isOpponent = !isOpponent;
            randomMove = getRandomAction(curPosition, emptyPoint, n);
        }
        return isOpponent;
    }

    Node* expand(Node* node, bool isOpponent) {
        board curPosition = node->position;
        board::point move = node->legal.back();
        node->legal.pop_back();
        curPosition.place(move);
        nodePool.push_back(Node(curPosition, engine));
        nodePool.back().fromWhichMove = move;
        node->mapActionToChild[move.i] = &nodePool.back();
        node->childs.push_back(&nodePool.back());
        return node->childs.back();
    }

    void update(Node* node, int result) {
        node->visitCount++;
        node->wins += result;
        for (int i = 0; i < (int)traverseHistory.size(); i++) {
            if (traverseHistory[i] && node->mapActionToChild[i] != NULL) {
                node->mapActionToChild[i]->raveCount++;
                node->mapActionToChild[i]->raveWins += result;
            }
        }
    }

    board::point getRandomAction(board position, std::vector<board::point>& empty, int n) {
        if (empty.empty()) {
            std::cerr << "getRandomAction error" << std::endl;
            exit(0);
        }
        int i = 0;
        while (i < n) {
            std::uniform_int_distribution<int> uniform(i, n - 1);
            int randomIndex = uniform(engine);
//            int randomIndex = (rand() % (n - i)) + i;
            if (position.place(empty[randomIndex]) == board::legal) {
                std::swap(empty[randomIndex], empty[n-1]);
                return empty[n-1];
            } else {
                std::swap(empty[randomIndex], empty[i]);
                i++;
            }
        }
        return empty[0];
    }

    bool isBlackTurn(bool isOpponent) {
        return (!isOpponent && who == board::black) || (isOpponent && who == board::white);
    }

    float uct(Node& node, int parentVisitCount, bool isOpponent) {
        if (node.visitCount == 0) return 10000.0;
//        float beta = sqrt(1 / (3 * parentVisitCount + 1));
        float beta = (float)node.raveCount /
                ((float)node.visitCount + (float)node.raveCount + 4 * (float)node.visitCount * (float)node.raveCount * 0.025 * 0.025);
        float winRate = (float)node.wins / (float)(node.visitCount + 1);
        float raveWinRate = (float)node.raveWins / (float)(node.raveCount + 1);
        // TODO: Need to think about anti uct
//        float exploitation = (isOpponent && uctType == "anti")? 1 - winRate : (1 - beta) * winRate + beta * raveWinRate;
        float exploitation = (isOpponent && uctType == "anti")?
                             (1 - beta) * (1 - winRate) + beta * (1 - raveWinRate) : (1 - beta) * winRate + beta * raveWinRate;
        float exploration = sqrt(log(parentVisitCount) / (float)(node.visitCount + 1));
        return exploitation + exploreC * exploration;
//        float score = exploitation + exploreC * exploration;
//        return (isOpponent && uctType == "anti")? -score : score;
    }

    board::point findActionByNextBoard(const board& nextBoard) {
        std::vector<board::point> temSpace = actions;
        for (board::point& move : temSpace) {
            board position = root->position;
            if (position.place(move) == board::legal && position == nextBoard)
                return move;
        }
        std::cerr << "find action error" << std::endl;
        exit(0);  // Error with call
    }

    std::string appendPath(std::string path, const action::place& move) {
        std::string moveCode = std::to_string(move.position().x) + std::to_string(move.position().y);
        return path + "_" + moveCode;
    }

    void cleanHistory() {
        for (int i = 0; i < (int)traverseHistory.size(); i++)
            traverseHistory[i] = 0;
    }

private:
    Node* root;
    float exploreC;
    std::vector<board::point> actions;
    std::vector<Node> nodePool;
    std::vector<int> traverseHistory;
    board::piece_type who;
    std::default_random_engine engine;
    std::uniform_int_distribution<int> uniform;
    std::string uctType;
};


#endif //TCG_NOGO_MCTS_MCTS_H
