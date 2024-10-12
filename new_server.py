import socket
import ssl
import time
from time import sleep

import yaml
import os
import pickle
import threading
import numpy as np
import torch
from Models import Mnist_2NN, Mnist_CNN, cifar10_cnn
from model.WideResNet import WideResNet
from copy import deepcopy
from numpy import *
import tenseal as ts
import struct

def serialize_encrypted_params(encrypted_params):
    serialized = {}
    for var in encrypted_params:
        serialized[var] = encrypted_params[var].serialize()
    return pickle.dumps(serialized)  # 使用 pickle 序列化整个字典




def tes_mkdir(path):

    if not os.path.isdir(path):
        os.mkdir(path)


def l2(parameters):  # 求l2范数的函数
    l2 = 0
    for var in parameters:
        l2 += torch.norm(parameters[var], p=2)
    return l2


def parameters_cosine(parameters1, parameters2):  # 定义求余弦相似度函数
    cosine = []
    for var in parameters2:
        cos = torch.mean(torch.cosine_similarity(parameters1[var], parameters2[var], dim=-1))
        cos = cos.cpu().numpy()
        cosine.append(deepcopy(cos))
        cos1 = sum(cosine)
    return cos1


def cosine_medain_sum(list):  # 定义求余弦相似度函数
    cosine_sum = 0
    for i in range(len(list)):
        if list[i] >= np.median(list):
            cosine_sum += list[i]
    return cosine_sum



def MPA(parameter):
    new_parameter = dict.fromkeys(parameter, 0)  # 加权聚合后的全局模型
    for var in parameter:
        new_parameter[var] = parameter[var].reshape(-1).to('cpu')
        new_parameter[var] = np.array(new_parameter[var])
        np.random.shuffle(new_parameter[var])
        new_parameter[var] = new_parameter[var].reshape(parameter[var].shape)
        new_parameter[var] = torch.from_numpy(new_parameter[var])
        new_parameter[var] = new_parameter[var].to('cuda')
    return new_parameter


class SSLServer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.server_socket = None
        self.client_list = {}  # 用于存储客户端编号和信息
        self.dev=torch.device("cpu") if self.config['is_cpu'] else torch.device("cuda")
        self.net=None
        self.model_weights=None
        self.num_clients_initialized = 0  # 跟踪初始化的客户端数量
        self.client_initialized_event = threading.Event()  # 用于同步客户端初始化的事件
        self.num_clients_initialized_lock = threading.Lock()
        self.client_encrypted_parameters_list = [0] *self.config['num_of_clients']  # 用于存储客户端加密的模型参数
        self.client_noised_parameters_list = [0] *self.config['num_of_clients']   # 用于存储客户端加噪音的模型参数
        self.num_sent_model_predictions = 0  # 跟踪发送的模型预测数量
        self.sent_model_predictions_event = threading.Event()  # 用于同步发送模型预测的事件
        self.num_sent_model_predictions_lock = threading.Lock()
        self.context=None
        self.num_send_enc_model=0
        self.send_enc_model_event=threading.Event()
        self.num_send_enc_model_lock=threading.Lock()
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['gpu']

        # mnist_2nn
        if self.config['model_name'] == 'mnist_2nn':
            self.net = Mnist_2NN()
        # mnist_cnn
        elif self.config['model_name'] == 'mnist_cnn':
            self.net = Mnist_CNN()
        # ResNet网络
        elif self.config['model_name'] == 'wideResNet':
            self.net = WideResNet(depth=28, num_classes=10)
        elif self.config['model_name'] == 'cifar10_cnn':
            self.net = cifar10_cnn()
        # 如果有多个GPU
        if self.config['is_cpu']:
            self.dev=torch.device("cpu")
            print("We use CPU!")
        elif torch.cuda.is_available():
            self.dev = torch.device("cuda")
            print("We use GPU!")
        else:
            self.dev = torch.device("cpu")
            print("With no GPU, CPU is the only option.")

        if torch.cuda.device_count() > 1 and self.dev==torch.device("cuda"):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = torch.nn.DataParallel(self.net)


        self.net = self.net.to(self.dev)

        #如果saved_model_parameters不为空，加载模型参数
        if self.config['saved_model_parameters']:
            self.net.load_state_dict(torch.load(self.config['saved_model_parameters']))

        self.model_weights = self.net.state_dict()

    def load_config(self,config_file):
        with open(config_file, 'r',encoding='utf-8') as file:
            return yaml.safe_load(file)

    def create_server_socket(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=self.config['certfile'], keyfile=self.config['keyfile'])
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.config['host'], self.config['port']))
        server_socket.listen(self.config['num_of_clients'])
        return context.wrap_socket(server_socket, server_side=True)

    def fetch_context(self):
        key_distribution_server_ip=self.config['key_distribution_server_ip']
        key_distribution_server_port=self.config['key_distribution_server_port']
        port_used_for_fetching_context=self.config['port_used_for_fetching_context']
        context = ssl._create_unverified_context()
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.bind(('localhost',port_used_for_fetching_context))
        sock.connect((key_distribution_server_ip,key_distribution_server_port))
        sock=context.wrap_socket(sock,server_hostname=key_distribution_server_ip)
        sock.sendall("Request for context".encode())
        size_data=sock.recv(4)
        size=struct.unpack('!I',size_data)[0]
        print(f"Expected size for context: {size}")
        context_bytes=b''
        while len(context_bytes)<size:
            chunk=sock.recv(min(4096,size-len(context_bytes)))
            if not chunk:
                raise RuntimeError("Socket connection broken")
            context_bytes+=chunk
        print("Context received.")
        self.context=ts.context_from(context_bytes)
        if self.context.is_private():
            print("Private context")
        else:
            print("Public context")
        sock.sendall("context received".encode())
        sock.close()


    def start(self):
        self.fetch_context()
        self.server_socket = self.create_server_socket()
        print(f"Server listening on {self.config['host']}:{self.config['port']}")
        threading.Thread(target=self.accept_clients, daemon=True).start()

        self.client_initialized_event.wait()
        if len(self.client_list) == self.config['num_of_clients']:
            print("All clients connected.\nModel training begins.")
            print('*'*100)
        else:
            print(f"Only {len(self.client_list)} clients connected, expected {self.config['num_of_clients']}. Terminating.")
            return
        num_in_comm = self.config['num_of_clients']  # 每轮参与通信的客户端数量

        for round in range(self.config['num_communication_rounds']):
            print(f"Round {round+1} begins.")
            # 为每个客户端发送模型参数
            for client_id in range(self.config['num_of_clients']):
                #新创建一个线程，处理每个客户端
                threading.Thread(target=self.receive_model_parameters, args=(client_id, self.client_list[client_id]['client_socket'])).start()
            self.sent_model_predictions_event.wait()

            enc_weight_dict=self.MAD()


            for i in range(self.config['num_of_clients']):
                #多线程发送
                threading.Thread(target=self.send_encrypted_model_parameters, args=(self.client_list[i]['client_socket'], enc_weight_dict)).start()
            self.send_enc_model_event.wait()
            #一些参数置为0
            self.num_sent_model_predictions = 0
            self.sent_model_predictions_event.clear()
            self.num_send_enc_model=0
            self.send_enc_model_event.clear()

            print(f"Round {round+1} ends.")

    def MAD(self):
        sum_cos=0
        similarity_list=[]
        for i in range(len(self.client_noised_parameters_list)):
            cos_i=[]
            for j in range(len(self.client_noised_parameters_list)):
                cos=parameters_cosine(self.client_noised_parameters_list[i],self.client_noised_parameters_list[j])
                cos_i.append(deepcopy(cos))
            cos_i=np.array(cos_i)
            similarity_list.append(cosine_medain_sum(cos_i))

        similarity_list=np.array(similarity_list)
        index=np.argsort(-similarity_list)
        for i in range(len(similarity_list)):
            if i<=int(len(similarity_list)/2):
                sum_cos+=similarity_list[index[i]]

        for i in range(len(similarity_list)):
            if i<=int(len(similarity_list)/2):
                similarity_list[index[i]]=similarity_list[index[i]]/sum_cos
            else:
                similarity_list[index[i]]=0
        print(f"这一轮的相似度列表为：{similarity_list}")
        weight_dic=dict.fromkeys(self.client_noised_parameters_list[0],0)
        for i in range(len(similarity_list)):
            for var in weight_dic:
                weight_dic[var]+=self.client_encrypted_parameters_list[i][var]*similarity_list[i]
        return weight_dic

    def send_encrypted_model_parameters(self, client_socket, encrypted_parameters):
        """
        发送加密的模型参数给客户端。

        :param client_socket: 客户端套接字
        :param encrypted_parameters: 加密的模型参数
        """
        try:
            # 发送即将发送的数据类型消息
            message = "Ready to send encrypted weights"
            client_socket.sendall(message.encode())
            print(f"Sent message: {message}")

            # 等待客户端确认
            confirmation = client_socket.recv(1024).decode()
            while confirmation != "Ready to receive":
                confirmation = client_socket.recv(1024).decode()
                print("Client is not ready, waiting...")
                time.sleep(0.5)

            # 序列化加密参数
            parameters_serialized = serialize_encrypted_params(encrypted_parameters)
            data_size = len(parameters_serialized)

            # 发送数据大小
            client_socket.sendall(struct.pack('!Q', data_size))
            print(f"Sent encrypted weights size: {data_size}")

            sent_bytes = 0
            while sent_bytes < data_size:
                sent_bytes += client_socket.send(parameters_serialized[sent_bytes:])
                print(f"Sent {sent_bytes} bytes of {data_size} bytes.")

            # 等待客户端确认接收
            confirmation = client_socket.recv(1024).decode()
            while confirmation != "Encrypted weights received":
                confirmation = client_socket.recv(1024).decode()
                print("Waiting for encrypted weights acknowledgment...")
                time.sleep(0.5)
            print("Encrypted weights acknowledged.")

            with self.num_send_enc_model_lock:
                self.num_send_enc_model += 1
                if self.num_send_enc_model == self.config['num_of_clients']:
                    self.send_enc_model_event.set()

        except Exception as e:
            print(f"Error sending encrypted weights: {e}")

    def accept_clients(self):
        while len(self.client_list) < self.config['num_of_clients']:
            client_socket, addr = self.server_socket.accept()
            client_id = len(self.client_list)
            self.client_list[client_id] = {"client_socket": client_socket, "addr": addr}
            print(f"Client {client_id} connected: {addr}")
            threading.Thread(target=self.initialize_client, args=(client_id, client_socket)).start()



    def initialize_client(self, client_id, client_socket):
        try:
            notification_message = "Please prepare to receive initial parameters."
            client_ack = self.send_notification_and_wait_for_ack(client_socket, notification_message)
            if client_ack is None:
                print(f"Client {client_id} did not acknowledge the notification correctly. Terminating connection.")
                return  # 终止处理此客户端
            # 发送初始参数
            initial_params = {
                'batch_size': self.config['batchsize'],
                'model_name': self.config['model_name'],
                'dataset': self.config['dataset'],
                'learning_rate': self.config['learning_rate'],
                'epoch': self.config['epoch'],
                'num_communication_rounds': self.config['num_communication_rounds'],
                'iid': self.config['IID'],
            }
            self.ensure_client_acknowledgement(client_socket, yaml.dump(initial_params))

            self.send_model_weights(client_socket, self.model_weights)
            with self.num_clients_initialized_lock:
                self.num_clients_initialized += 1
                if self.num_clients_initialized == self.config['num_of_clients']:
                    self.client_initialized_event.set()  # 所有客户端都初始化完成，设置事件

            # 这里可以添加更多的处理逻辑，例如发送训练数据等
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")

    def send_notification_and_wait_for_ack(self, client_socket, notification, max_attempts=5):
        """
        发送通知并等待客户端确认，具有最大尝试次数。
        
        :param client_socket: 客户端套接字
        :param notification: 要发送的通知消息
        :param max_attempts: 最大尝试次数
        :return: 客户端确认消息，如果未收到确认则为None
        """
        for attempt in range(max_attempts):
            client_socket.sendall(notification.encode())
            print(f"Sent notification: {notification} (Attempt {attempt + 1}/{max_attempts})")

            try:
                client_ack = client_socket.recv(1024).decode()
                print(f"Received acknowledgement: {client_ack} (Attempt {attempt + 1})")
                if client_ack == "Ready to receive parameters.":
                    return client_ack
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

            print(f"Client did not acknowledge the notification correctly, attempt {attempt + 1}/{max_attempts}. Resending...")
        
        print(f"Failed to receive correct acknowledgement after {max_attempts} attempts.")
        return None
        

    def ensure_client_acknowledgement(self, client_socket, message):
        """
        确保客户端确认接收指定的消息。
        
        :param client_socket: 客户端套接字
        :param message: 要发送的消息
        """
        try:
            print("Sending initial_parameters to client:")
            while True:
                client_socket.sendall(message.encode('utf-8'))
                confirmation = client_socket.recv(1024).decode()
                if confirmation == "Parameters acknowledged":
                    print("Client acknowledged the message.")
                    break
                else:
                    print("Client did not acknowledge the message, resending...")
        except Exception as e:
            print(f"Error sending initial parameters: {e}")

    def send_model_weights(self, client_socket, model_weights, max_attempts=5):
        """
        将模型权重序列化并发送给客户端，直到客户端确认或达到最大尝试次数。

        :param client_socket: 客户端套接字
        :param model_weights: 模型权重
        :param max_attempts: 最大尝试次数
        """
        # 序列化模型权重的大小
        weights_size = len(pickle.dumps(model_weights))
        weights_size_encoded = pickle.dumps(weights_size)

        # 发送模型权重的大小
        client_socket.sendall(weights_size_encoded)
        print("Sent model weights size.")

        # 等待客户端确认接收模型权重的大小
        confirmation = client_socket.recv(1024).decode()
        while confirmation != "Size acknowledged":
            confirmation = client_socket.recv(1024).decode()
            print("Client did not acknowledge size, retrying...")

        # 发送模型权重
        for attempt in range(max_attempts):
            try:
                # 序列化模型权重
                weights_encoded = pickle.dumps(model_weights)
                client_socket.sendall(weights_encoded)
                print(f"Sent model weights (Attempt {attempt + 1}/{max_attempts})")

                # 等待客户端确认接收模型权重
                confirmation = client_socket.recv(1024).decode()
                if confirmation == "Weights acknowledged":
                    print("Model weights sent and acknowledged.")
                    return
                else:
                    print(f"Client did not acknowledge weights, attempt {attempt + 1}. Resending...")
            except Exception as e:
                print(f"Error sending model weights on attempt {attempt + 1}: {e}")

        print("Failed to send model weights after maximum attempts.")

    def receive_model_parameters(self, index:int, client_socket):
        """
        从客户端接收数据，并确保每次传输的成功进行。

        :param client_socket: 客户端套接字
        """
        noised_weights = None
        encrypted_weights = None
        try:
            for data_name in ("noised_weights", "encrypted_weights"):
                # 接收客户端即将发送的数据类型消息
                message = client_socket.recv(1024).decode()
                while message != f"Ready to send {data_name}":
                    message = client_socket.recv(1024).decode()
                    print(f"Received message from client: {message}")
                    time.sleep(0.5)

                # 向客户端确认服务器已准备好接收数据
                client_socket.sendall("Ready to receive".encode())

                # 接收数据大小
                size_data = client_socket.recv(8)
                data_size = struct.unpack('!Q', size_data)[0]
                print(f"Expected size for {data_name}: {data_size}")

                # 接收数据
                data_serialized = b''
                while len(data_serialized) < data_size:
                    chunk = client_socket.recv(min(1048576, data_size - len(data_serialized)))
                    if not chunk:
                        raise RuntimeError("Socket connection broken")
                    data_serialized += chunk
                print(f"{data_name} received.")

                # 解码数据
                if data_name == "noised_weights":
                    noised_weights = pickle.loads(data_serialized)
                    print(f"Received noised weights from client{index}")
                elif data_name == "encrypted_weights":
                    encrypted_weights = pickle.loads(data_serialized)
                    for var in encrypted_weights:
                        encrypted_weights[var] = ts.ckks_vector_from(self.context,encrypted_weights[var])
                    print(f"Received encrypted weights from client{index}")

                client_socket.sendall(f"{data_name} received".encode())

                # 存储接收到的参数
            self.client_encrypted_parameters_list[index] = encrypted_weights
            self.client_noised_parameters_list[index] = noised_weights
            with self.num_sent_model_predictions_lock:
                self.num_sent_model_predictions += 1
                if self.num_sent_model_predictions == self.config['num_of_clients']:
                    self.sent_model_predictions_event.set()
        except Exception as e:
            print(f"Connection error: {e}. Closing connection.")
            client_socket.close()
            return None


    


if __name__ == "__main__":
    server = SSLServer('./server.config.yaml')
    server.start()