from trajectory_socket import TrajectoryServer
import trajectory_pb2 as trajectory


def self_play_loop(actor_socket: TrajectoryServer):
    while True:
        byte_int = actor_socket.recv(4)
        size = 0
        for i in range(4):
            size *= pow(2, 8)
            size += int(byte_int[i])
        
        raw = actor_socket.recv(size)
        print("receive raw data len: {}".format(len(raw)))
        parsed_trajectory = trajectory.trajectory()
        parsed_trajectory.ParseFromString(raw)
        print("num_transition: ", len(parsed_trajectory.transitions))
        print("last reward: ", parsed_trajectory.transitions[-1].reward)


if __name__ == "__main__":
    learner_server = TrajectoryServer('0.0.0.0', 7000)
    learner_server.start()
    self_play_loop(learner_server)
    