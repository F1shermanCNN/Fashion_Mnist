1. 项目简介
本项目基于 **FashionMNIST** 数据集，旨在通过构建深度学习模型实现对 10 类服装图像的自动分类。

2. 环境配置
本项目使用 Python 开发，需要安装对应依赖。
# 安装依赖
pip install -r requirements.txt

3. 训练脚本
在本项目中可以通过training.complete_training.py文件进行训练，训练前先设置好超参数列表即可开始完整训练，训练结束后会自动保存最优参数、损失历史图像、精确率变化图像等在本地。
调整后在根目录运行
python -m training.complete_training

4. 测试脚本
本项目中可以通过testing.test_input_params.py文件进行参数测试，首先下载好最优参数到根目录，本项目附带了best_params_1.npz，可以自行更改，将路径改为下载好的参数文件的路径，随后根据文件形式修改隐藏层大小等超参数。
可以返回在测试集上的准确率以及混淆矩阵
调整后在根目录运行
python -m testing.test_input_params



