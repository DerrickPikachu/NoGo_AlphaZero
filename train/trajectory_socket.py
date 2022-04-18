import socket
import trajectory_pb2 as trajectory


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
        if self.connected_sock is None:
            connection, address = self.trajectory_sock.accept()
            print('Connect with actor ' + str(address))
            self.connected_sock = connection
            self.actor_addr = address
            
    def send(self, data):
        if self.connected_sock is not None:
            self.connected_sock.send(data.encode())
            
    def recv(self, size):
        if self.connected_sock is not None:
            return self.connected_sock.recv(size)
        return None
        

if __name__ == "__main__":
    server = TrajectoryServer('0.0.0.0', 7000)
    server.start()
    byte_int = server.recv(4)
    size = 0
    for i in range(4):
        size *= pow(2, 8)
        size += int(byte_int[i])
    raw = server.recv(size)
    print("receive raw data len: {}".format(len(raw)))
    parsed_trajectory = trajectory.trajectory()
    parsed_trajectory.ParseFromString(raw)
    print("num_transition: ", len(parsed_trajectory.transitions))
    print("last reward: ", parsed_trajectory.transitions[-1].reward)
