# eye/zmq_tools.py

import zmq
import threading
import pickle
import numpy as np
import torch
class Publisher:
    def __init__(self, bind_address,conflate=True):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(bind_address)

        # Set High Water Mark to 1
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)

        # Set Conflate option to true
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)

        self.socket.bind(bind_address)

    def send_string(self, message):
        self.socket.send_string(message)

    def send_bytes(self, data):
        self.socket.send(data, flags=zmq.NOBLOCK)

    def send_pyobj(self, obj):
        self.socket.send_pyobj(obj)

    def send_json(self, json_obj):
        self.socket.send_json(json_obj)

    def send_action_chunk(self, action_chunk, timestamp):
        # First convert from bfloat16 to float32, then to numpy
        np_array = action_chunk.detach().to(torch.float32).cpu().numpy()
        timestamp = np.float64(timestamp)
        shape = np.array(np_array.shape, dtype=np.int32)

        # Send three parts: timestamp, array, and shape
        self.socket.send_multipart([
            timestamp.tobytes(),
            np_array.tobytes(),
            shape.tobytes(),
        ])

class Subscriber:
    def __init__(self, connect_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(connect_address)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.callbacks = {"string": None, "bytes": None, "pyobj": None, "json": None, "action_chunk": None}
        self.running = False

    def on_recv_string(self, callback):
        self.callbacks["string"] = callback
        self._start_receiving()

    def on_recv_bytes(self, callback):
        self.callbacks["bytes"] = callback
        self._start_receiving()

    def on_recv_pyobj(self, callback):
        self.callbacks["pyobj"] = callback
        self._start_receiving()

    def on_recv_json(self, callback):
        self.callbacks["json"] = callback
        self._start_receiving()

    def on_recv_action_chunk(self, callback):
        self.callbacks["action_chunk"] = callback
        self._start_receiving()

    def _start_receiving(self):
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self._receive_loop)
            thread.daemon = True
            thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                
                if self.callbacks["string"]:
                    message = self.socket.recv()
                    try:
                        string_message = message.decode("utf-8")
                        self.callbacks["string"](string_message)
                    except UnicodeDecodeError:
                        pass

                if self.callbacks["bytes"]:
                    message = self.socket.recv()
                    self.callbacks["bytes"](message)

                if self.callbacks["pyobj"]:
                    message = self.socket.recv()
                    try:
                        obj = pickle.loads(message)
                        self.callbacks["pyobj"](obj)
                    except pickle.UnpicklingError:
                        pass

                if self.callbacks["json"]:
                    try:
                        json_obj = self.socket.recv_json()
                        self.callbacks["json"](json_obj)
                    except zmq.ZMQError:
                        pass

                if self.callbacks["action_chunk"]:
                    try:
                        raw_parts = self.socket.recv_multipart()
                        if len(raw_parts) == 3:  # We expect 2 parts now: timestamp and array
                            timestamp_bytes, array_bytes, shape_bytes = raw_parts
                            # Convert timestamp
                            timestamp = np.frombuffer(timestamp_bytes, dtype=np.float64)[0]
                            shape = np.frombuffer(shape_bytes, dtype=np.int32)
                            # Convert array (assuming shape is (1, 1, 60, 7) from the original code)
                            action_chunk = np.frombuffer(array_bytes, dtype=np.float32).reshape(shape)
                            
                            # Call the callback
                            self.callbacks["action_chunk"](action_chunk, timestamp)
                    except Exception as e:
                        pass

            except zmq.ZMQError as e:
                print(f"ZMQ Error: {e}")
                break

        self.socket.close()
        self.context.term()
