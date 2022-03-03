import socket


class TrajectoryServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.trajectory_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.trajectory_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.trajectory_sock.bind((self.host, self.port))
        self.trajectory_sock.listen(1)
        
    def start(self):
        connection, address = self.trajectory_sock.accept()
        print('Connect with actor ' + str(address))
        while True:
            data = connection.recv(1024)
            print(data.decode())
            connection.send(data)
        

if __name__ == "__main__":
    server = TrajectoryServer('0.0.0.0', 7000)
    server.start()

# HOST = '0.0.0.0'
# PORT = 7000

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# s.bind((HOST, PORT))
# s.listen(1)

# while True:
#     conn, addr = s.accept()
#     print('connect by ' + str(addr))
#     input()