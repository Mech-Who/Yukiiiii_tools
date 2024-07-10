# 个人工具库

## 一、介绍

    个人用库，汇集各类工具以及自己的工具代码。便于快速安装个性化的依赖。

## 二、安装

    ```shell
    # 用https://pypi.org/simple/，避免镜像源没有及时更新
    pip install -i https://pypi.org/simple/ yukiiiii_tools
    ```

## 三、用法示例

    1. 四则运算：

        ```python
        from yukiiiii_tools.calculate import add, subtract, multiply, divide
        print(add(2, 2))
        # >>> 4
        print(subtract(2, 2))
        # >>> 0
        print(multiply(2, 2))
        # >>> 4
        print(divide(2, 2))
        # >>> 1
        ```

    2. 模型可视化

        ```python
        from yukiiiii_tools.visulize import show_model_in_netron

        model_path="/path/to/onnx/model/file"
        show_model_in_netron(model_path)
        # 启动一个服务器用于可视化模型结构
        ```

    3. 记录评价指标

        ```python
        from yukiiiii_tools.utils import AverageMeter

        loss_meter = AverageMeter()
        print(loss_meter.val, loss_meter.count, loss_meter.sum, loss_meter.avg)
        # >>> 0 0 0 0
        loss_meter.update(1.5)
        print(loss_meter.val, loss_meter.count, loss_meter.sum, loss_meter.avg)
        # >>> 1.5 1 1.5 1.5
        loss_meter.update(2, 2)
        print(loss_meter.val, loss_meter.count, loss_meter.sum, loss_meter.avg)
        # >>> 2 3 5.5 1.8333333333333333
        loss_meter.reset()
        print(loss_meter.val, loss_meter.count, loss_meter.sum, loss_meter.avg)
        # >>> 0 0 0 0
        ```

## 四、相关依赖

    - netron：7.7.4

## 五、开发环境说明

    - Python版本：3.10
    - 使用`conda`管理依赖，保证开发环境一致性
    - 初始化项目：`pip install -r requirements.txt`安装依赖
    - 使用netron实现可视化onnx存储的模型
    - 使用Github Actions实现自动化部署

## 六、修订记录

- 2024.07.10：添加依赖库ptflops，用于统计pytorch模型的计算量和参数量。
- 2024.07.08：创建 AverageMeter 工具类，主要用于深度学习记录评价指标值。
- 2024.06.24：引入 netron 库，用于可视化模型结构。
