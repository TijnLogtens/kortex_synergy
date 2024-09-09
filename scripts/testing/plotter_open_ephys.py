import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zmq
import numpy as np
import json
import uuid
import time
from threading import Thread, Lock

class PlotProcess:
    def __init__(self):
        self.context = zmq.Context()
        self.data_socket = None
        self.event_socket = None
        self.poller = zmq.Poller()
        self.message_num = -1
        self.socket_waits_reply = False
        self.app_name = 'Plot Process'
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.channel_num = 128  # Assuming 128 channels for EMG
        self.sample_rate = 30000  # Assuming a sample rate of 30000 Hz
        self.num_samples_per_message = 640  # Based on your observation
        self.buffer_size = 10 * self.sample_rate  # Buffer to hold 10 seconds of data
        self.data_buffer = np.zeros((self.channel_num, self.buffer_size))
        self.buffer_index = 0
        self.channel_data_received = [False] * self.channel_num
        self.running = True
        self.lines = None
        self.fig = None
        self.ax = None
        self.lock = Lock()  # To synchronize access to shared resources

    def init_socket(self):
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect("tcp://localhost:5556")
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.poller.register(self.data_socket, zmq.POLLIN)

        self.event_socket = self.context.socket(zmq.REQ)
        self.event_socket.connect("tcp://localhost:5557")
        self.poller.register(self.event_socket, zmq.POLLIN)

    def send_heartbeat(self):
        while self.running:
            try:
                if (time.time() - self.last_heartbeat_time) > 2.0:
                    d = {'application': self.app_name, 'uuid': self.uuid, 'type': 'heartbeat'}
                    j_msg = json.dumps(d)
                    print("sending heartbeat")
                    self.event_socket.send(j_msg.encode('utf-8'))
                    self.last_heartbeat_time = time.time()
                    self.socket_waits_reply = True

                if self.socket_waits_reply:
                    socks = dict(self.poller.poll(1000))
                    if self.event_socket in socks:
                        message = self.event_socket.recv()
                        print(f'Heartbeat reply: {message}')
                        self.socket_waits_reply = False
            except zmq.ZMQError as err:
                print(f"Heartbeat error: {err}")
                time.sleep(1)

    def start(self):
        self.init_socket()
        Thread(target=self.send_heartbeat).start()
        Thread(target=self.callback).start()
        self.init_plot()

    def init_plot(self):
        self.fig, self.ax = plt.subplots()
        self.lines = [self.ax.plot(np.zeros(self.buffer_size))[0] for _ in range(self.channel_num)]
        self.ax.set_ylim([-250, 250])
        self.ax.set_xlim([0, self.buffer_size])
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

    def update_plot(self, frame):
        with self.lock:
            for line, channel_data in zip(self.lines, self.data_buffer):
                line.set_ydata(np.roll(channel_data, -self.buffer_index))
            self.ax.set_xlim(0, self.buffer_size)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def callback(self):
        while self.running:
            socks = dict(self.poller.poll(1))
            if not socks:
                continue

            if self.data_socket in socks:
                try:
                    message = self.data_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as err:
                    print(f"got error: {err}")
                    continue

                if message:
                    try:
                        header = json.loads(message[1].decode('utf-8'))
                    except ValueError as e:
                        print(f"ValueError: {e}")
                        print(message[1])
                        continue

                    if header['type'] == 'data':
                        c = header['content']
                        channel_num = c['channel_num']
                        num_samples = c['num_samples']

                        if num_samples != self.num_samples_per_message:
                            print(f"Unexpected number of samples: {num_samples}")
                            continue

                        try:
                            data = np.frombuffer(message[2], dtype=np.float32)
                            with self.lock:
                                start_index = self.buffer_index % self.buffer_size
                                end_index = (self.buffer_index + num_samples) % self.buffer_size

                                if start_index < end_index:
                                    self.data_buffer[channel_num, start_index:end_index] = data
                                else:
                                    self.data_buffer[channel_num, start_index:] = data[:self.buffer_size - start_index]
                                    self.data_buffer[channel_num, :end_index] = data[self.buffer_size - start_index:]

                                self.buffer_index += num_samples
                                self.channel_data_received[channel_num] = True

                                if all(self.channel_data_received):
                                    self.channel_data_received = [False] * self.channel_num
                        except ValueError as e:
                            print(e)
                            print(header)
                            print(message[1])

    def stop(self):
        self.running = False

if __name__ == '__main__':
    plotter = PlotProcess()
    try:
        plotter.start()
    except KeyboardInterrupt:
        plotter.stop()
