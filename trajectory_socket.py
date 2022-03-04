import socket


class TrajectoryServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.trajectory_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.trajectory_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.trajectory_sock.bind((self.host, self.port))
        self.trajectory_sock.listen(1)
        self.connected_sock = None
        self.actor_addr = None
        
    def start(self):
        if connected_sock is None:
            connection, address = self.trajectory_sock.accept()
            print('Connect with actor ' + str(address))
            self.connected_sock = connection
            self.actor_addr = address
            # while True:
            #     data = connection.recv(1024)
            #     print(data.decode())
            #     connection.send(data)
            
    def send(data):
        if connected_sock is not None:
            self.connected_sock.send(data.encode())
            
    def recv():
        if connected_sock is not None:
            return self.connected_sock.recv(1024).decode()
        return None
        

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