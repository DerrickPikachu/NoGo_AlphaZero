#include "trajectory_socket.h"


TrajectorySocket::TrajectorySocket() {
    server_host = "127.0.0.1";
    server_port = 8787;
}

void TrajectorySocket::setup_socket() {
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr(server_host.c_str());
    server_address.sin_port = server_port;
}