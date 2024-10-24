### Federal-Learning-Attack-Detection

### 环境

1.python3.8

2.pytorch1.7.1

3.pip install -r requirements.txt

### 使用

其中new_client.py是客户端主程序，加载了client.config.yaml  
可以配置服务器IP端口等，同时还可以设定投毒方式，目前只支持模型投毒和标签反转  
new_server.py为服务器主程序，配置文件为server.config.yaml  
可以配置数据集，模型，batch_size等超参数  
key_distribution_server.py为密钥分发服务器，配置文件为key_distribution_server.py  
在开启环境前，请先将密钥分发服务器运行起来，后续聚合服务器和客户端都将从密钥服务器拿CKKS.context  
开启后服务器将处于监听状态，直到有指定数目的客户端主动连接才会进行训练

### config参数说明
- client
  ```
    host: "127.0.0.1"      #聚合服务器IP  
    port: 12360            #聚合服务器端口  
    certfile: "./cert.pem"     #可以指定聚合服务器本地证书位置  
    attack: NA      #attack为攻击方式label flipping attack:LFA  //model poisoning attacks:MPA   //no attack:NA  
    path_to_save: ./checkpoints      #保存训练模型权重位置  
    key_distribution_server_ip: "127.0.0.1"    #密钥分发服务器IP  
    key_distribution_server_port: 6653          #密钥分发服务器端口  
    key_distribution_server_certfile: "./cert.pem"  ##可以指定密钥分发服务器本地证书位置  
  ```  
- server
  ```
  host: "127.0.0.1"  # 服务器的IP地址
    port: 12360       # 服务器监听的端口号
    key_distribution_server_ip: "127.0.0.1"    #密钥分发服务器IP 
    key_distribution_server_port: 6653          #密钥分发服务器端口
    port_used_for_fetching_context: 22226       #连接密钥分发服务器端口
    certfile: "cert.pem"  # SSL证书文件的路径
    keyfile: "key.pem"   # SSL私钥文件的路径
    is_cpu: false       # 是否使用CPU
    gpu: "0"            # 使用的GPU ID
    num_of_clients: 4  # 客户端的数量
    cfraction: 1.0      # 随机挑选的客户端的比例
    epoch: 1            # 客户端本地训练的迭代次数
    batchsize: 16       # 客户端本地训练的批次大小
    model_name: "mnist_cnn"  # 训练模型的名称   可选：mnist_cnn，cifar10_cnn，wideResNet，mnist_2nn
    learning_rate: 0.0002     # 学习率
    dataset: "mnist"        # 使用的数据集      可选：mnist，cifar10
    save_freq: 100            # 全局模型保存的频率（通信次数）
    num_communication_rounds: 70             # 通信次数（训练轮次）
    saved_model_parameters: ""  # 加载的模型参数路径,如果为空则不加载
    IID: 1                    # 数据分配方式（IID数据分布）
    algorithm: "APFL"         # 使用的联邦学习算法
    pr: 0.0                   # 污染数据的比例
  ```

- key_distribution_server
  ```
    host: "127.0.0.1"  # 密钥分发服务器的IP地址
    port: 6653        # 密钥分发服务器监听的端口号
    server_ip: "127.0.0.1"  # 服务器的IP地址
    server_port: 22226  # 服务器用于获取上下文的端口号
    certfile: "./cert.pem"  # SSL证书文件的路径
    keyfile: "./key.pem"  # SSL私钥文件的路径
  ```


### 文件和目录结构
| 文件/目录                       | 描述                 |
| -----------------              | ---------            |
| `data/`                        | 数据集目录            |
| `checkpoints/`                 | checkpointer保存路径  |
| `model/`                       | 模型配置文件          |
| `new_client.py`                | 客户端                |
| `getData.py`                   | 配置数据集            |
| `Models.py`                    | 模型配置              |
| `requirements.txt`             | 项目依赖              |
| `new_server.py`                | 服务器端              |
| `key_distribution_server.py`   | 密钥分发服务器端       |
