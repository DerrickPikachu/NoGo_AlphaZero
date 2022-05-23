#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <exception>
#include <tuple>

#include "board.h"
#include "action.h"


class AlphaZeroException : public std::exception {
public:
    AlphaZeroException(std::string msg) :
        std::exception(),
        err_msg(msg) {}
    
    const char* what() const noexcept override {
        return err_msg.c_str();
    }

private:
    std::string err_msg;
};

class WritePipeException : public AlphaZeroException {
public:
    WritePipeException(std::string msg) :
        AlphaZeroException(msg) {}
};

class ReadPipeException : public AlphaZeroException {
public:
    ReadPipeException(std::string msg) :
        AlphaZeroException(msg) {}
};

class ExpandException : public AlphaZeroException {
public:
    ExpandException(std::string msg) :
        AlphaZeroException(msg) {}
};


const std::vector<std::string> split(const std::string &str, const char &delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string tok;

    while (std::getline(ss, tok, delimiter)) {
        result.push_back(tok);
    }
    return result;
}


class PipeInterface {
public:
    virtual ~PipeInterface() {}
    virtual bool write_to_pipe(std::string data) = 0;
    virtual std::string read_from_pipe() = 0;
    virtual void close_write() = 0;
    virtual void close_read() = 0;
    virtual void redirect_stdout() = 0;
    virtual void redirect_stdin() = 0;
};


class Pipe : public PipeInterface {
public:
    Pipe() {
        int p[2];
        if (pipe(p) == -1)
            std::cerr << "pipe create error" << std::endl;
        pipe_read = p[0];
        pipe_write = p[1];
        write_active = true;
        read_active = true;
    }

    bool write_to_pipe(std::string data) override {
        if (!write_active) {
            throw WritePipeException("Write after close write");
        }
        write(pipe_write, data.c_str(), data.size());
    }

    std::string read_from_pipe() override {
        if (!read_active) {
            throw ReadPipeException("Read after close read");
        }
        std::string forward_result = "";
        char buffer[1024];
        int read_byte;
        while ((read_byte = read(pipe_read, buffer, 1023)) >= 0) {
            buffer[read_byte] = '\0';
            forward_result += std::string(buffer);
            if (forward_result.back() == '#') {
                forward_result.pop_back();
                break;
            }
        }
        if (read_byte < 0) {
            std::cerr << "read error" << std::endl;
            exit(-1);
        }
        return forward_result;
    }

    void close_write() override {
        close(pipe_write);
        write_active = false;
    }

    void close_read() override {
        close(pipe_read);
        read_active = false;
    }

    void redirect_stdout() override {
        dup2(pipe_write, STDOUT_FILENO);
    }

    void redirect_stdin() override {
        dup2(pipe_read, STDIN_FILENO);
    }

protected:
    int pipe_write;
    int pipe_read;
    bool write_active;
    bool read_active;
};


class NetInterface {
public:
    virtual ~NetInterface() {}
    virtual void exec_net() = 0;
    virtual std::string get_forward_result(const board&) = 0;
    virtual void refresh_model(std::string model_name) = 0;
    virtual void send_exit() = 0;
    virtual std::pair<std::vector<float>, float> parse_result(std::string) = 0; 
};


class AlphaZeroNet : public NetInterface {
public:
    // TODO: Here should be config
    AlphaZeroNet(
        std::string net_path, std::string weight_path, int board_size) :
        path_to_net(net_path),
        path_to_weight(weight_path),
        board_size(board_size),
        write_pipe(NULL),
        read_pipe(NULL) {}

    void exec_net() override {
        write_pipe = new Pipe();
        read_pipe = new Pipe();
        int process_id = fork();
        if (process_id == -1) {
            std::cerr << "create alphazero net error" << std::endl;
            exit(1);
        } else if (process_id > 0) {
            write_pipe->close_read();
            read_pipe->close_write();
        } else {
            write_pipe->close_write();
            read_pipe->close_read();
            write_pipe->redirect_stdin();
            read_pipe->redirect_stdout();
            char** args = build_args();
            if (execvp("python", args) == -1) {
                std::cerr << "exec alphazero net error" << std::endl;
                exit(1);
            }
            std::cerr << "fail" << std::endl;
        }
    }

    std::string get_forward_result(const board& state) override {
        write_pipe->write_to_pipe("forward\n");
        std::string encoded_state = board_to_string(state);
        write_pipe->write_to_pipe(encoded_state + "\n");
        return read_pipe->read_from_pipe();
    }

    void refresh_model(std::string name) override {
        write_pipe->write_to_pipe("refresh\n");
        write_pipe->write_to_pipe(name + "\n");
    }

    void send_exit() override {
        write_pipe->write_to_pipe("exit\n");
        write_pipe->close_write();
        read_pipe->close_read();
        delete write_pipe;
        delete read_pipe;
    }

    std::pair<std::vector<float>, float> parse_result(std::string result) override {
        std::vector<std::string> policy_value = split(result, ';');
        std::vector<std::string> policy = split(policy_value[0], ',');
        std::vector<float> float_policy;
        float_policy.reserve(policy.size());
        for (auto& prob_str : policy) {
            float_policy.push_back(std::stof(prob_str));
        }
        return {float_policy, std::stof(policy_value[1])};
    }
    
private:
    char** build_args() {
        // python path/to/alphazero_net.py -p path/to/weight -bs 9
        std::vector<std::string> args_arr({
            "python",
            path_to_net,
            "-p",
            path_to_weight,
            "-bs",
            std::to_string(board_size)
        });
        char** args = new char*[args_arr.size() + 1];
        args[args_arr.size()] = NULL;
        for (int i = 0; i < args_arr.size(); i++) {
            char* char_arg = new char[args_arr[i].size()];
            strcpy(char_arg, args_arr[i].c_str());
            args[i] = char_arg;
        }
        return args;
    }

    std::string board_to_string(const board& state) {
        std::string encoded_board;
        for (int x = 0; x < board::size_x; x++) {
            for (int y = 0; y < board::size_y; y++) {
                if (state[x][y] == 2u) {
                    encoded_board += "-1.0";
                } else if (state[x][y] == 3u || state[x][y] == 0u) {
                    encoded_board += "0.0";
                } else if (state[x][y] == -1u) {
                    std::cerr << "board with unknow piece type" << std::endl;
                    exit(1);
                } else if (state[x][y] == 1u) {
                    encoded_board += "1.0";
                }
                encoded_board += ",";
            }
        }
        // remove redundent common
        encoded_board.pop_back();
        return encoded_board;
    }

private:
    std::string path_to_net;
    std::string path_to_weight;
    int board_size;

protected:
    PipeInterface* write_pipe;
    PipeInterface* read_pipe;
};


class NodeInterface {
public:
    ~NodeInterface() {}
    virtual NodeInterface* select() = 0;
    virtual float expand(NetInterface*) = 0;
    virtual void update(float value) = 0;
    virtual float value() = 0;
    virtual void reset() = 0;
    virtual board::point best_action() = 0;
    virtual board get_state() = 0;
    virtual board::piece_type get_color() = 0;
    virtual bool expanded() = 0;
};

class Node : public NodeInterface {
public:
    Node(const board& b, board::piece_type piece) :
        value_sum(0.0),
        visit_count(0),
        state(b),
        piece_color(piece),
        is_expand(false) {}

    ~Node() = default;

    NodeInterface* select() override {
        if (!is_expand)
            throw AlphaZeroException("Select error: node hasn't been expand");
        if (childs.empty())
            return nullptr;
        float max_score = 0.0;
        Node* best_node;
        for (int i = 0; i < childs.size(); i++) {
            Node& child = std::get<2>(childs[i]);
            float score = puct(childs[i]);
            if (max_score < score) {
                max_score = score;
                best_node = &child;
            }
        }
        return best_node;
    }

    float expand(NetInterface* net) override {
        if (is_expand) {
            throw ExpandException(
                "Expand error: The node has been expanded, but try to expand again");
        }
        int action_space = board::size_x * board::size_y;
        board::piece_type child_color = 
            (piece_color == board::piece_type::black)? 
            board::piece_type::white : board::piece_type::black;
        std::vector<board::point> valid_actions = get_valid_actions();
        if (valid_actions.empty()) {  // end state
            return (child_color == board::piece_type::black)?
                1.0 : -1.0;
        }
        std::string result = net->get_forward_result(state);
        std::pair<std::vector<float>, float> policy_value = 
            net->parse_result(result);
        for (int i = 0; i < valid_actions.size(); i++) {
            int action_id = valid_actions[i].i;
            board tem = state;
            tem.place(valid_actions[i]);
            childs.push_back({
                policy_value.first[action_id],
                valid_actions[i],
                Node(tem, child_color)
            });
        }
        is_expand = true;
        return policy_value.second;
    }

    void update(float value) override {
        value_sum += value;
        visit_count++;
    }
    
    void reset() override {
        value_sum = 0.0;
        visit_count = 0;
        childs.clear();
        state = board();
    }

    float value() override {
        if (visit_count == 0)
            return 0.0;
        return value_sum / visit_count;
    }

    board::point best_action() override {
        int max_visit_count = 0;
        board::point best_point;
        for (int i = 0; i < childs.size(); i++) {
            Node& child_node = std::get<2>(childs[i]);
            if (child_node.get_visit_count() > max_visit_count) {
                max_visit_count = child_node.get_visit_count();
                best_point = std::get<1>(childs[i]);
            }
        }
        return best_point;
    }

    board get_state() override { return state; }
    board::piece_type get_color() override { return piece_color; }
    bool expanded() override { return is_expand; }
    int get_visit_count() { return visit_count; }

private:
    float puct(std::tuple<float, board::point, Node>& child) {
        float prob = std::get<0>(child);
        Node& child_node = std::get<2>(child);
        float prior = 
            prob * (float)std::sqrt(visit_count) 
            / (child_node.get_visit_count() + 1);
        float value = (piece_color == board::piece_type::black)?
            child_node.value() : -child_node.value();
        // std::cout << "value: " << value << "\tprior: " << prior << std::endl;
        return value + prior;
    }

    std::vector<board::point> get_valid_actions() {
        int action_space = board::size_x * board::size_y;
        std::vector<board::point> valid_actions;
        valid_actions.reserve(action_space);
        for (int i = 0; i < action_space; i++) {
            board tem = state;
            board::point action(i);
            if (tem.place(action) == board::legal) {
                valid_actions.push_back(action);
            }
        }
        return valid_actions;
    }

protected:
    std::vector<std::tuple<float, board::point, Node>> childs;
    bool is_expand;

private:
    float value_sum;
    int visit_count;
    board state;
    board::piece_type piece_color;
};

class TreeInterface {
public:
    ~TreeInterface() {}
    virtual void select() = 0;
    virtual float expand() = 0;
    virtual void update(float) = 0;
    virtual board::point get_action() = 0;
    virtual void set_root(NodeInterface*) = 0;
};

class Tree : public TreeInterface {
public:
    Tree(NetInterface* network_provider) :
        net(network_provider),
        root(nullptr),
        select_node(nullptr) {}
    ~Tree() = default;

    void select() override {
        if (!root->expanded())
            throw AlphaZeroException("Select error: root haven't been expanded");
        NodeInterface* next_node = root;
        history.push_back(next_node);
        while (next_node->expanded()) {
            try {
                next_node = next_node->select();
            } catch (std::exception& c) {
                throw;
            }
            if (next_node == nullptr)  break;
            history.push_back(next_node);
        }
        select_node = history.back();
    }

    float expand() override {
        if (select_node == nullptr)
            throw AlphaZeroException("Expand error: tree select node is null");
        return select_node->expand(net);
    }

    void update(float winrate) override {
        if (history.empty())
            throw AlphaZeroException("Update error: the history is empty");
        for (NodeInterface* node : history) {
            node->update(winrate);
        }
    }

    board::point get_action() override { return root->best_action(); }
    void set_root(NodeInterface* node) { root = node; }

protected:
    NodeInterface* root;
    NodeInterface* select_node;
    std::vector<NodeInterface*> history;

private:
    NetInterface* net;
};