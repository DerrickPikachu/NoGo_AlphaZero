#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <exception>

#include "board.h"


class WritePipeException : public std::exception {
public:
    WritePipeException(std::string msg) :
        std::exception(),
        err_msg(msg) {}

    const char* what() const noexcept override {
        return err_msg.c_str();
    }

private:
    std::string err_msg;
};


class ReadPipeException : public std::exception {
public:
    ReadPipeException(std::string msg) :
        std::exception(),
        err_msg(msg) {}

    const char* what() const noexcept override {
        return err_msg.c_str();
    }

private:
    std::string err_msg;
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
        std::string forward_result;
        char buffer[820];
        int read_byte;
        if ((read_byte = read(pipe_read, buffer, 819)) < 0) {
            std::cerr << "read error" << std::endl;
            exit(1);
        }
        buffer[read_byte] = '\0';
        forward_result = std::string(buffer);
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
    virtual void refresh_model() = 0;
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

    void refresh_model() override {
        write_pipe->write_to_pipe("refresh\n");
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