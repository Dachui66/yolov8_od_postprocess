# yolov8目标检测模型部署
本工程为yolov8目标检测模型的后处理c++实现，并演示了如何使用python调用c++模块，验证后处理功
能的正确性，以便将后处理模块部署在嵌入式或边缘设备上, 最终完成yolov8目标检测模型的部署。

## 环境搭建
OS: ubuntu18.04/ubuntu20.04测试通过。
- 源码编译并安装opencv3.4.5
- 安装Anaconda
- conda create -n yolov8 python=3.9 -y
- conda activate yolov8
- pip install -r requirements.txt 

## 编译
- git clone https://github.com/Dachui66/yolov8_od_postprocess.git
- cd yolov8_od_postprocess
- 修改compile.sh, PYBIND11_PATH=/path/to/anaconda3/envs/yolov8/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
- ./compile.sh

## 运行
- ./run.sh

## 结果
- 查看当前目录下生成的res.jpg是否正确框出目标。


