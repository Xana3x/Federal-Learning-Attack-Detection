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
