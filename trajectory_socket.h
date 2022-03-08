#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>

class TrajectorySocket {
public:
    TrajectorySocket();
    TrajectorySocket(std::string host, int port);

private:
    void setup_socket();

private:
    int socket_fd;
    int server_port;
    std::string server_host;
    struct sockaddr_in server_address;
};