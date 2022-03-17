#include <iostream>
#include <fstream>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <ext/stdio_filebuf.h>

class SocketInterface {
public:
    virtual ~SocketInterface() {};
    virtual bool connect_server() = 0;
    virtual std::string receive() = 0;
    virtual void send(std::string data) = 0;
};

class TrajectorySocket : public SocketInterface {
public:
    TrajectorySocket() {
        server_host = "127.0.0.1";
        server_port = 8787;
        setup_socket();
    };
    ~TrajectorySocket() {};
    TrajectorySocket(std::string host, int port) {
        server_host = host;
        server_port = port;
        setup_socket();
    }

    bool connect_server() override {
        int result = connect(socket_fd, (struct sockaddr *)&server_address, sizeof(server_address));
        return result == 0;
    }

    std::string receive() override {
        memset(buffer, 0, sizeof(buffer));
        int size = read(socket_fd, buffer, sizeof(buffer));
        buffer[size] = 0;
        return std::string(buffer);
    }

    void send(std::string data) override {
        int result = write(socket_fd, data.c_str(), data.size());
    }

private:
    void setup_socket() {
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        server_address.sin_family = AF_INET;
        server_address.sin_addr.s_addr = inet_addr(server_host.c_str());
        server_address.sin_port = server_port;
        __gnu_cxx::stdio_filebuf<char> filebuf(socket_fd, std::ios::in);
    }

protected:
    int socket_fd;
    int server_port;
    std::string server_host;
    struct sockaddr_in server_address;
    char buffer[1024];
};