import io
import math
import socket
import ssl
import threading
import yaml
import warnings  # 屏蔽warning
import struct
import pickle
warnings.filterwarnings("ignore", message="WARNING: The input does not fit in a single ciphertext*")
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, cifar10_cnn
from model.WideResNet import WideResNet
import time
from numpy import *
import tenseal as ts
from getData import GetDataSet
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
import copy


MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB

def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass

def enc(parameters, context):
    encrypted_params = {}
    for var in parameters:
        param = parameters[var].flatten().cpu().numpy()  # 转换为 numpy 数组
        encrypted_params[var] = ts.ckks_vector(context, param)
    return encrypted_params

def serialize_encrypted_params(encrypted_params):
    serialized = {}
    for var in encrypted_params:
        serialized[var] = encrypted_params[var].serialize()
    return pickle.dumps(serialized)  # 使用 pickle 序列化整个字典

def dec(parameters, secret_key):  # 解密模型参数
    parameters1 = dict.fromkeys(parameters, 0)  # 重新定一个字典存储模型参数，不要改变原parameters的值，不然后面的客户端的parameters是解密之后的值
    for var in parameters:
        parameters1[var] = parameters[var].decrypt(secret_key)
    return parameters1

def send_in_chunks(socket, data):
    """Send data in chunks of 10MB."""
    chunk_size = 5 * 1024 * 1024  # 10MB
    total_size = len(data)
    socket.sendall(struct.pack('!Q', total_size))  # Send total data size

    # Send the data in chunks
    for i in range(0, total_size, chunk_size):
        chunk = data[i:i + chunk_size]
        socket.sendall(chunk)
        time.sleep(1)
    print(f"Sent data in chunks, total size: {total_size}")

def receive_in_chunks(socket, total_size):
    """Receive data in chunks of 10MB."""
    chunk_size = 5 * 1024 * 1024  # 10MB
    received_data = b''

    while len(received_data) < total_size:
        remaining_size = total_size - len(received_data)
        bytes_to_receive = min(chunk_size, remaining_size)
        chunk = socket.recv(bytes_to_receive)
        if not chunk:
            return None
        received_data += chunk

    print(f"Received data in chunks, total size: {total_size}")
    return received_data


class SSLClient:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.key_distribution_socket=None
        self.client_socket = None
        self.parameters = None
        self.loss_func= F.cross_entropy
        self.opti = None
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.dev=torch.device("cpu")
        self.net=None
        #self.dataset=None
        self.train_data=None
        self.train_label=None
        self.IID=None
        self.train_ds=None
        self.train_dl=None
        self.test_ds=None
        self.test_dl=None
        self.test_data=None
        self.test_label=None
        self.context = None
        self.secret_key = None

    def MPA(self,parameter):
        new_parameter = dict.fromkeys(parameter, 0)  # 加权聚合后的全局模型
        for var in parameter:
            new_parameter[var] = parameter[var].reshape(-1).to(self.dev)
            new_parameter[var] = np.array(new_parameter[var])
            np.random.shuffle(new_parameter[var])
            new_parameter[var] = new_parameter[var].reshape(parameter[var].shape)
            new_parameter[var] = torch.from_numpy(new_parameter[var])
            new_parameter[var] = new_parameter[var].to(self.dev)
        return new_parameter


    def fetch_key(self):
        """
        连接到密钥分发服务器并接收密钥。

        :return: 密钥, 如果接收失败则返回None
        """
        try:
            host = self.config['key_distribution_server_ip']
            port = self.config['key_distribution_server_port']
            certfile = self.config.get('key_distribution_server_certfile')
            #context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=certfile)
            context = ssl._create_unverified_context()
            self.key_distribution_socket = socket.create_connection(('localhost', port))
            self.key_distribution_socket = context.wrap_socket(self.key_distribution_socket, server_hostname=host)
            print(f"Connected to key server at {host}:{port}")

            # 接收密钥长度
            context_length_data = self.key_distribution_socket.recv(4)
            context_length = struct.unpack('!I', context_length_data)[0]
            print(f"Received context length: {context_length}")

            # 接收字节流
            context_bytes = b''
            while len(context_bytes) < context_length:
                data = self.key_distribution_socket.recv(1024)
                if not data:
                    break
                context_bytes += data

            # 加载 context
            self.context = ts.context_from(context_bytes)
            if self.context.is_private():
                self.secret_key = self.context.secret_key()
                print("Context and secret key received successfully.")
            else:
                print("Received context does not contain a secret key.")

            # 发送确认
            self.key_distribution_socket.sendall("Key acknowledged".encode())
            return print(type(self.context))
        except Exception as e:
            print(f"Error connecting to key server: {e}")
            return None


    def load_config(self, config_file):
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)

    def create_client_socket(self):
        '''
        创建客户端socket
        param: host 主机地址
        param: port 端口
        param: certfile 证书文件路径
        '''
        host = self.config['host']
        port = self.config['port']
        certfile = self.config.get('certfile')  # 如果没有提供证书文件，则默认为None
        #context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=certfile)
        context = ssl._create_unverified_context()
        client_socket = socket.create_connection((host, port))
        return context.wrap_socket(client_socket, server_hostname=host)

    def connect(self):
        '''
        连接到服务器
        '''
        self.client_socket = self.create_client_socket()    
        buffer_size=200*1024*1024
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        print(f"Connected to {self.config['host']}:{self.config['port']}")

    def receive_notification(self):
        while True:
            notification_message = self.client_socket.recv(1024).decode()
            if notification_message=="Please prepare to receive initial parameters.":
                print(f"Received notification: {notification_message}")
                self.client_socket.sendall("Ready to receive parameters.".encode())
                return
            else:
                print("Waiting for notification...")

    def receive_parameters(self):
        # 初始化参数数据

        # 循环接收数据直到接收完毕
        try:
            while True:
                # 接收数据片段
                params_data = b''
                data = self.client_socket.recv(1024)
                #params_data += data

                # 尝试解析参数
                try:
                    self.parameters = yaml.safe_load(data.decode())
                    print(f"Received initial parameters: {self.parameters}")
                    self.client_socket.sendall("Parameters acknowledged".encode())
                    if self.parameters['model_name'] == 'mnist_2nn':
                        self.net = Mnist_2NN()
                    # mnist_cnn
                    elif self.parameters['model_name'] == 'mnist_cnn':
                        self.net = Mnist_CNN()
                    # ResNet网络
                    elif self.parameters['model_name'] == 'wideResNet':
                        self.net = WideResNet(depth=28, num_classes=10)
                    elif self.parameters['model_name'] == 'cifar10_cnn':
                        self.net = cifar10_cnn()
                    # 如果有多个GPU

                    # if torch.cuda.device_count() > 1:
                    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
                    #     self.net = torch.nn.DataParallel(self.net)

                    self.net = self.net.to(self.dev)

                    self.dataset = GetDataSet(self.parameters['dataset'], self.parameters['iid'])
                    self.train_data = self.dataset.train_data
                    self.train_label = self.dataset.train_label
                    self.test_data=self.dataset.test_data
                    self.test_label=self.dataset.test_label

                    if self.parameters['dataset'] == 'cifar10':
                        self.opti = optim.Adam(self.net.parameters(), lr=self.parameters['learning_rate'])
                        print("优化算法：", self.opti)
                    elif self.parameters['dataset'] == 'mnist':
                        self.train_label = torch.argmax(torch.tensor(self.train_label), dim=1)
                        self.opti = optim.SGD(self.net.parameters(), lr=self.parameters['learning_rate'], momentum=0.9)
                        print("优化算法：", self.opti)

                    
                    
                    self.train_ds = TensorDataset(torch.tensor(self.train_data), torch.tensor(self.train_label))
                    self.train_dl = DataLoader(self.train_ds, batch_size=self.parameters['batch_size'], shuffle=True)
                    self.test_ds = TensorDataset(torch.tensor(self.test_data), torch.tensor(self.test_label))
                    self.test_dl = DataLoader(self.test_ds, batch_size=100, shuffle=False)
                    

                    self.client_socket.sendall("Parameters acknowledged".encode())
                    return
                except yaml.YAMLError:
                    print("Received incomplete parameters. Waiting for more data...")
        except Exception as e:
            print(f"Error receiving initial parameters: {e}")
            return None

    def run(self):
        self.fetch_key()
        self.connect()
        try:
            self.receive_notification()
            self.receive_parameters()

            model_weights = self.receive_model_weights(self.client_socket)
            if model_weights:
                self.net.load_state_dict(model_weights)
                print("Model weights loaded successfully.")
            else:
                print("Failed to load model weights.")
            
            for round in range(self.parameters['num_communication_rounds']):
                print(f"Starting communication round {round + 1}.")

                state_dict_enc, state_dict_Gaussian=self.local_update()
                
                if self.config['attack']=='MPA':
                    state_dict_Gaussian=self.MPA(state_dict_Gaussian)

                self.send_data_to_server(self.client_socket,  state_dict_Gaussian, state_dict_enc)
                self.receive_encrypted_model_weights(self.client_socket)
                print(f"Communication round {round + 1} completed.")
            print("All communication rounds completed.")
            makedir(self.config['path_to_save'])
            torch.save(self.net.state_dict(), f"{self.config['path_to_save']}/{self.parameters['model_name']}_"
                                              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}.pth")
        except Exception as e:
            print(f"ERROR with function run: {e}")
        finally:
            self.client_socket.close()
            print("Connection closed.")
        
    
    def recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    
    def receive_large_data(self, client_socket, num_connections, total_size):
        ports = []
        sockets = []
        for _ in range(num_connections):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            ports.append(port)
            s.listen(1)
            sockets.append(s)
        
        # 发送端口号列表给客户端
        client_socket.sendall(struct.pack(f'!{num_connections}H', *ports))
        
        data = [b"" for _ in range(num_connections)]
        threads = []

        def receive_chunk(index):
            conn, addr = sockets[index].accept()
            #print(f"Connection from {addr} for chunk {index}")
            try:
                # 接收 "Ready to send" 消息
                message = conn.recv(1024).decode()
                #print(f"Received message: {message}")
                if message != "Ready to send":
                    raise ValueError(f"Unexpected message: {message}")
                
                # 发送 "Ready to receive" 消息
                conn.sendall("Ready to receive".encode())
                
                # 接收数据大小
                size_data = conn.recv(8)
                chunk_size = struct.unpack('!Q', size_data)[0]
                #print(f"Expected chunk size: {chunk_size}")
                
                # 确认接收到大小信息
                conn.sendall("Size received".encode())
                
                # 接收数据
                received_data = b''
                while len(received_data) < chunk_size:
                    chunk = conn.recv(min(MAX_CHUNK_SIZE, chunk_size - len(received_data)))
                    if not chunk:
                        raise ConnectionError("Connection closed before receiving all data")
                    received_data += chunk
                
                #print(f"Received {len(received_data)} bytes for chunk {index}")
                
                # 存储接收到的数据
                data[index] = received_data
                
                # 发送确认消息
                conn.sendall("Data received".encode())
            
            except Exception as e:
                print(f"Error in receiving chunk {index}: {e}")
            finally:
                conn.close()
            

        for i in range(num_connections):
            thread = threading.Thread(target=receive_chunk, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for s in sockets:
            s.close()

        return b''.join(data)[:total_size]

    def recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data


    def receive_encrypted_model_weights(self, client_socket):
        """
        接收服务器发送的加密模型权重。
        
        :param client_socket: 客户端套接字
        :return: 加密模型权重,如果接收失败则返回None
        """
        try:
            # 接收服务器即将发送的数据类型消息
            message = client_socket.recv(1024).decode()
            while message != "Ready to send encrypted weights":
                message = client_socket.recv(1024).decode()
                print(f"Received message from server: {message}")
                time.sleep(0.5)

            # 向服务器确认客户端已准备好接收数据
            client_socket.sendall("Ready to receive".encode())

            # 接收键的数量
            num_keys_data = client_socket.recv(4)
            num_keys = struct.unpack('!I', num_keys_data)[0]
            #print(f"Expected number of keys: {num_keys}")

            # 存储接收的加密模型权重
            received_parameters = {}

            for _ in range(num_keys):
                # 接收键
                key_size_data = client_socket.recv(8)
                key_size = struct.unpack('!Q', key_size_data)[0]
                key_serialized = self.recv_all(client_socket, key_size)
                key = pickle.loads(key_serialized)

                # 接收值大小
                value_size_data = client_socket.recv(8)
                value_size = struct.unpack('!Q', value_size_data)[0]
                #print(f"Receiving value of size {value_size} for key {key}")

                # 接收连接数量
                num_connections_data = client_socket.recv(4)
                num_connections = struct.unpack('!I', num_connections_data)[0]

                if num_connections > 1:
                    # 为大型数据创建新的监听端口
                    value_serialized = self.receive_large_data(client_socket, num_connections, value_size)
                else:
                    # 接收小型数据
                    value_serialized = self.recv_all(client_socket, value_size)
                
                # 解密并处理模型权重
                value = ts.ckks_vector_from(self.context, value_serialized)
                received_parameters[key] = value

                # 向服务器确认接收了一对键值对
                client_socket.sendall("Pair received".encode())

            print("Encrypted weights received successfully.")

            # 确认接收完毕
            client_socket.sendall("Encrypted weights received".encode())
            
            # 将接收到的加密权重存储到模型中
            for key, value in received_parameters.items():
                self.net.state_dict()[key].data = torch.tensor(value.decrypt())
            
            self.net.to(self.dev)

            return received_parameters

        except Exception as e:
            print(f"Error receiving encrypted weights: {e}")
            return None

    def process_model_prameters(self,model_parameters_encrypted):
        model_parameters_dec = dec(model_parameters_encrypted, self.secret_key)
        for var in model_parameters_dec:
            model_parameters_dec[var] = np.array(model_parameters_dec[var]).reshape(self.net.state_dict()[var].shape)
            model_parameters_dec[var] = torch.tensor(model_parameters_dec[var], dtype=torch.float32, device=torch.device(self.dev))
    

    def local_update(self):
        # 传入网络模型，并加载global_parameters参数的
        flip_probability = 0.5  # Probability of flipping a label
        num_classes = 10  # Assuming MNIST or CIFAR-10
        # 载入Client自有数据集
        # 加载本地数据
        #self.train_dl = DataLoader(self.train_ds, batch_size=self.parameters['batch_size'], shuffle=True)
        def flip_label(label):
            if random.random() < flip_probability:
                return random.randint(0, num_classes - 1)
            return label
        # 设置迭代次数
        for epoch in range(self.parameters['epoch']):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                if self.config['attack']=='LFA':
                    label = flip_label(label)
                preds = self.net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                if self.parameters['dataset'] == 'cifar10':
                    label = label.long()

                loss = self.loss_func(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                self.opti.step()
                # 将梯度归零，初始化梯度
                self.opti.zero_grad()
        # loss_txt.write('loss: ' + str(loss) + "\n")

        def l2(parameters):  # 求l2范数的函数
            l2 = 0
            for var in parameters:
                l2 += torch.norm(parameters[var], p=2)
            return l2

        # 对局部模型参数添加高斯噪声
        def addnoice(C, epsilon, delta, model_parameters,dataset_name,dev):
            c = np.sqrt(2 * 100 * np.log(1/delta))
            if dataset_name == 'cifar10':
                s = 2*C/2380
            else:
                s = 2*C/2857
            l2 = 0
            for var in model_parameters:
                l2 += torch.norm(model_parameters[var], p=2)
            for var in model_parameters:
                model_parameters[var] = model_parameters[var] / max(1, l2/C)  # 裁剪，将l2范数限制在C以内
                model_parameters[var] = model_parameters[var]+torch.normal(0, ((c**2) * (s**2))/(epsilon * epsilon), model_parameters[var].shape).to(dev)
            return model_parameters
        # Net.state_dict_Gaussian = addnoice(l2(global_parameters), 0.01, 40, Net.state_dict())  # 设置高斯噪声的参数并加噪

        for var in self.net.state_dict():  # 裁剪阈值为40
            self.net.state_dict()[var] = self.net.state_dict()[var] / max(1, l2(self.net.state_dict())/40)
        
        
        state_dict = copy.deepcopy(self.net.state_dict())
        state_dict_Gaussian = addnoice(40, 60, 0.01,  state_dict,self.parameters['dataset'],self.dev)  # 设置高斯噪声的参数并加噪

        
        state_dict = enc(state_dict, self.context)
        print("encrpyt completed!!!!")
        # Net.state_dict_Gaussian = dict.fromkeys(Net.state_dict(), 0)  # 初始化加噪后的局部模型参数
        return state_dict, state_dict_Gaussian  # 返回本地训练的模型参数Net.state_dict()和高斯加噪版Net.state_dict_Gaussian

    def send_data_to_server(self, main_socket, noised_weights, encrypted_weights):
        for data, data_name in zip((noised_weights, encrypted_weights), ("noised_weights", "encrypted_weights")):
            # 发送特殊消息告知服务器即将传输的数据类型
            message = f"Ready to send {data_name}"
            main_socket.sendall(message.encode())
            #print(f"Sent message: {message}")

            # 等待服务器确认
            confirmation = main_socket.recv(1024).decode()
            while confirmation != "Ready to receive":
                confirmation = main_socket.recv(1024).decode()
                print(f"Server is not ready for {data_name}, waiting...")
                time.sleep(0.5)

            # 发送键的数量
            num_keys = len(data)
            main_socket.sendall(struct.pack('!I', num_keys))
            #print(f"Sent number of keys: {num_keys}")

            # 逐个发送键值对
            for key, value in data.items():
                # 序列化键
                key_serialized = pickle.dumps(key)
                key_size = len(key_serialized)
                main_socket.sendall(struct.pack('!Q', key_size))
                main_socket.sendall(key_serialized)

                # 序列化值
                if data_name == "encrypted_weights":
                    value_serialized = value.serialize()
                else:
                    value_serialized = pickle.dumps(value)
                value_size = len(value_serialized)
                main_socket.sendall(struct.pack('!Q', value_size))

                num_connections = math.ceil(value_size / MAX_CHUNK_SIZE)
                main_socket.sendall(struct.pack('!I', num_connections))

                if num_connections > 1:
                    # 接收新的端口号列表
                    ports = struct.unpack(f'!{num_connections}H', main_socket.recv(2 * num_connections))
                    
                    # 发送大型数据
                    self.send_large_data(value_serialized, ('127.0.0.1', ports))
                else:
                    # 发送小型数据
                    main_socket.sendall(value_serialized)

                # 等待服务器确认接收
                confirmation = main_socket.recv(1024).decode()
                while confirmation != "Pair received":
                    confirmation = main_socket.recv(1024).decode()
                    print(f"Waiting for pair acknowledgment...")
                    time.sleep(0.5)

            #print(f"Sent {data_name}.")

            # 等待服务器确认接收所有数据
            confirmation = main_socket.recv(1024).decode()
            while confirmation != f"{data_name} received":
                confirmation = main_socket.recv(1024).decode()
                print(f"Waiting for {data_name} acknowledgment...")
                time.sleep(0.5)
            #print(f"{data_name} acknowledged.")

        print("All model parameters sent and acknowledged.")

    def send_large_data(self, data, address):
        sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(len(address[1]))]
        try:
            for i, s in enumerate(sockets):
                s.connect((address[0], address[1][i]))
                start = i * MAX_CHUNK_SIZE
                end = min((i + 1) * MAX_CHUNK_SIZE, len(data))
                chunk = data[start:end]
                chunk_size = len(chunk)
                
                # 发送 "Ready to send" 消息
                s.sendall("Ready to send".encode())
                
                # 等待 "Ready to receive" 消息
                message = s.recv(1024).decode()
                #print(f"Received message: {message}")
                if message != "Ready to receive":
                    raise ValueError(f"Unexpected message: {message}")
                
                # 发送数据大小
                s.sendall(struct.pack('!Q', chunk_size))
                
                # 等待确认接收到大小信息
                message = s.recv(1024).decode()
                if message != "Size received":
                    raise ValueError(f"Unexpected message: {message}")
                
                # 发送数据
                s.sendall(chunk)
                #print(f"Sent data chunk {i + 1}/{len(address[1])} ({chunk_size} bytes)")
                
                # 等待确认接收到数据
                message = s.recv(1024).decode()
                if message != "Data received":
                    raise ValueError(f"Unexpected message: {message}")
        
        except Exception as e:
            print(f"Error in sending data: {e}")
        finally:
            for s in sockets:
                s.close()

    def receive_model_weights(self, client_socket):
        """
        接收服务器发送的模型权重。

        :param client_socket: 客户端套接字
        :return: 模型权重,如果接收失败则返回None
        """
        try:
            # 接收模型权重的大小
            weights_size_encoded = client_socket.recv(1024)
            if not weights_size_encoded:
                print("No data received for weights size.")
                return None
            weights_size = pickle.loads(weights_size_encoded)
            print(f"Received model weights size: {weights_size}")

            # 确认接收完毕模型权重的大小
            client_socket.sendall("Size acknowledged".encode())
            print("start receiving weights")

            # 接收模型权重数据
            weights_encoded = b''
            while len(weights_encoded) < weights_size:
                data = client_socket.recv(1024)
                if not data:
                    break
                weights_encoded += data

            # 确认接收完毕并解码模型权重
            model_weights = pickle.loads(weights_encoded)
            print("Model weights received successfully.")
            client_socket.sendall("Weights acknowledged".encode())
            return model_weights

        except Exception as e:
            print(f"Error receiving model weights: {e}")
            return None
        
    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

if __name__ == "__main__":
    client = SSLClient('./client.config.yaml')
    client.run()
    client.test()
