import io
import socket
import ssl
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



    pass

class SSLClient:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.key_distribution_socket=None
        self.client_socket = None
        self.parameters = None
        self.loss_func= F.cross_entropy
        self.opti = None
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net=None
        #self.dataset=None
        self.train_data=None
        self.train_label=None
        self.IID=None
        self.train_ds=None
        self.train_dl=None
        self.context = None
        self.secret_key = None


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

                    if torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        self.net = torch.nn.DataParallel(self.net)

                    self.net = self.net.to(self.dev)

                    self.dataset = GetDataSet(self.parameters['dataset'], self.parameters['iid'])
                    self.train_data = self.dataset.train_data
                    self.train_label = self.dataset.train_label

                    if self.parameters['dataset'] == 'cifar10':
                        self.opti = optim.Adam(self.net.parameters(), lr=self.parameters['learning_rate'])
                        print("优化算法：", self.opti)
                    elif self.parameters['dataset'] == 'mnist':
                        self.train_label = torch.argmax(torch.tensor(self.train_label), dim=1)
                        self.opti = optim.SGD(self.net.parameters(), lr=self.parameters['learning_rate'], momentum=0.9)
                        print("优化算法：", self.opti)

                    self.train_ds = TensorDataset(torch.tensor(self.train_data), torch.tensor(self.train_label))
                    self.train_dl = DataLoader(self.train_ds, batch_size=self.parameters['batch_size'], shuffle=True)

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

                state_dict_enc, state_dict_Gaussian=self.local_update(self.net.state_dict())
                #print("!!!!!!!!!!!",state_dict_Gaussian)
                self.send_data_to_server(self.client_socket,  state_dict_Gaussian, state_dict_enc )
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

            # 接收数据大小
            size_data = client_socket.recv(8)
            data_size = struct.unpack('!Q', size_data)[0]
            print(f"Expected size for encrypted weights: {data_size}")

            # 接收数据
            weights_encoded = b''
            while len(weights_encoded) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(weights_encoded)))
                if not chunk:
                    raise RuntimeError("Socket connection broken")
                weights_encoded += chunk

            # 解码模型权重
            model_weights = pickle.loads(weights_encoded)

            print("Model weights received successfully.")

            # 确认接收完毕
            client_socket.sendall("Encrypted weights received".encode())

            # 处理接收到的加密权重
            for var in model_weights:
                model_weights[var] = ts.ckks_vector_from(self.context, model_weights[var])
                self.net.state_dict()[var].data = torch.from_numpy(model_weights[var].decrypt()).to(self.dev)

            return model_weights

        except Exception as e:
            print(f"Error receiving encrypted weights: {e}")
            return None

    def process_model_prameters(self,model_parameters_encrypted):
        model_parameters_dec = dec(model_parameters_encrypted, self.secret_key)
        for var in model_parameters_dec:
            model_parameters_dec[var] = np.array(model_parameters_dec[var]).reshape(self.net.state_dict()[var].shape)
            model_parameters_dec[var] = torch.tensor(model_parameters_dec[var], dtype=torch.float32, device=torch.device(self.dev))
    

    def local_update(self, model_parameters):
        # 传入网络模型，并加载global_parameters参数的
        self.net.load_state_dict(model_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        #self.train_dl = DataLoader(self.train_ds, batch_size=self.parameters['batch_size'], shuffle=True)

        # 设置迭代次数
        for epoch in range(self.parameters['epoch']):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
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
        state_dict=self.net.state_dict()
        state_dict_Gaussian = addnoice(40, 60, 0.01,  state_dict,self.parameters['dataset'],self.dev)  # 设置高斯噪声的参数并加噪

        for var in self.net.state_dict():  # 裁剪阈值为40
            self.net.state_dict()[var] = self.net.state_dict()[var] / max(1, l2(self.net.state_dict())/40)
        state_dict = enc(state_dict, self.context)
        print("encrpyt completed!!!!")
        # Net.state_dict_Gaussian = dict.fromkeys(Net.state_dict(), 0)  # 初始化加噪后的局部模型参数
        return state_dict, state_dict_Gaussian  # 返回本地训练的模型参数Net.state_dict()和高斯加噪版Net.state_dict_Gaussian

    def send_data_to_server(self, client_socket, noised_weights, encrypted_weights):
        for data, data_name in zip((noised_weights, encrypted_weights), ("noised_weights", "encrypted_weights")):
            # 发送特殊消息告知服务器即将传输的数据类型
            message = f"Ready to send {data_name}"
            client_socket.sendall(message.encode())
            print(f"Sent message: {message}")

            # 等待服务器确认
            confirmation = client_socket.recv(1024).decode()
            while confirmation != "Ready to receive":
                confirmation = client_socket.recv(1024).decode()
                print(f"Server is not ready for {data_name}, waiting...")
                time.sleep(0.5)

            # 序列化数据
            if data_name == "encrypted_weights":
                data_serialized = serialize_encrypted_params(data)
            else:
                data_serialized = pickle.dumps(data)

            data_size = len(data_serialized)

            # 发送数据大小
            client_socket.sendall(struct.pack('!Q', data_size))
            print(f"Sent {data_name} size: {data_size}")

            sent_bytes=0
            while sent_bytes < data_size:
                sent_bytes += client_socket.send(data_serialized[sent_bytes:])
                print(f"Sent {sent_bytes} bytes of {data_size} bytes.")

            print(f"Sent {data_name}.")

            # 等待服务器确认接收
            confirmation = client_socket.recv(1024).decode()
            while confirmation != f"{data_name} received":
                confirmation = client_socket.recv(1024).decode()
                print(f"Waiting for {data_name} acknowledgment...")
                time.sleep(0.5)
            print(f"{data_name} acknowledged.")

        print("All model parameters sent and acknowledged.")

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

if __name__ == "__main__":
    client = SSLClient('./client.config.yaml')
    client.run()
