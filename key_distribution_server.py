import socket
import ssl
import tenseal as ts
import yaml
import struct

class KeyDistributionServer:
    def __init__(self):
        self.config = self.load_config('key_distribution_server.yaml')
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2 ** 40
        self.context_bytes = self.context.serialize(save_secret_key=True)
        self.sock = self.create_server_socket()
        self.context_without_secret_key = self.context.copy()
        self.context_without_secret_key.make_context_public()
        self.context_without_secret_key_bytes = self.context_without_secret_key.serialize(save_secret_key=False)


    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def create_server_socket(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=self.config['certfile'], keyfile=self.config['keyfile'])
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.config['host'], self.config['port']))
        server_socket.listen()
        return context.wrap_socket(server_socket, server_side=True)

    def listen(self):
        print("Server is listening...")
        while True:
            conn, addr = self.sock.accept()
            if addr==(self.config['server_ip'], self.config['server_port']):
                print(f"Connection from {addr}")
                self.handle_server(conn)
            else:
                print(f"Connection from {addr}")
                self.handle_client(conn)

    def handle_client(self, conn):
        try:
            # 发送密钥长度
            context_length = len(self.context_bytes)
            conn.sendall(struct.pack('!I', context_length))
            print(f"Sent context length: {context_length}")

            # 发送字节流
            conn.sendall(self.context_bytes)
            print("Context sent successfully.")

            # 等待客户端确认
            acknowledgment = conn.recv(1024).decode()
            if acknowledgment == "Key acknowledged":
                print("Acknowledgment received. Closing connection.")
        except Exception as e:
            print(f"Error sending context: {e}")
        finally:
            conn.close()
            print("Connection closed.")

    def handle_server(self, conn):
        try:
            # 接收服务器的请求
            request = conn.recv(1024).decode()
            while request!="Request for context":
                request = conn.recv(1024).decode()
                print("Received request for context.")
            # 发送密钥长度
            conn.sendall(struct.pack('!I', len(self.context_without_secret_key_bytes)))

            # 发送密钥
            conn.sendall(self.context_without_secret_key_bytes)
            print("Aggregation context")

            # 等待服务器确认
            acknowledgment = conn.recv(1024).decode()
            while acknowledgment != "context received":
                acknowledgment = conn.recv(1024).decode()
                print("Received NOT acknowledgment.")
            print("Acknowledgment received. Closing connection.")
        except Exception as e:
            print(f"Error in server communication: {e}")
        finally:
            conn.close()
            print("Server connection closed.")

if __name__ == "__main__":
    server = KeyDistributionServer()
    server.listen()