# 个人工具库

## 一、介绍

    个人用库，汇集各类工具以及自己的工具代码。便于快速安装个性化的依赖。

## 二、安装

    ```shell
    # 用https://pypi.org/simple/，避免镜像源没有及时更新
    pip install -i https://pypi.org/simple/ yukiiiii_tools
    ```

## 三、用法示例

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

## 四、相关依赖

## 五、开发环境说明

    - Python版本：3.10
    - 使用`conda`管理依赖，保证开发环境一致性
    - 初始化项目：`pip install -r requirements.txt`安装依赖
