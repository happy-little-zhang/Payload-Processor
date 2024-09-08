import numpy as np
import csv
import seaborn as sns
import pandas as pd
import os
import re

# 将car_hacking正常驾驶数据集从txt转换成csv
def car_hacking_dataset_transform():
    read_path = "dataset/car_hacking/"
    file_name = "normal_run_data.txt"
    dataset_path = os.path.join(read_path, file_name)
    print(f"read path: {dataset_path}")

    csv_save_file = "attack_free.csv"
    save_path = os.path.join(read_path, csv_save_file)
    print(f"save_path: {save_path}")

    # 读取 TXT 文件
    with open(dataset_path, 'r') as file:
        data_str = file.read()

    # 初始化空列表,用于存储数据
    timestamps = []
    ids = []
    dlcs = []
    payloads = []

    # 按行分割数据
    lines = data_str.strip().split('\n')

    # 遍历每一行,提取数据
    for line in lines[:-1]:  # 遍历除最后一行之外的所有行,最后一行数据破损
        print(line)

        # 提取时间戳，单位转化为us
        timestamp = int(float(line.split('Timestamp: ')[1].split(' ')[0]) * 1000000)
        timestamps.append(timestamp)

        # 提取 ID
        id_str = line.split('ID: ')[1].split(' ')[0]
        #id_int = int(id_str, 16)
        ids.append(id_str)

        # 提取 DLC
        dlc = int(line.split('DLC: ')[1].split(' ')[0])
        dlcs.append(dlc)

        # 提取 Payload 并填充为 8 个字节
        #print(line.split(' '))
        #print(line.split('DLC: ')[1].split(' ')[4:])
        payload_str = line.split('DLC: ')[1].split(' ')[4:]
        #payload_bytes = [int(x, 16) for x in payload_str[0:dlc]]
        #payload_bytes = payload_bytes + ["00"] * (8 - len(payload_bytes))  # 填充为 8 个字节
        payload_bytes = payload_str + ["00"] * (8 - len(payload_str))  # 填充为 8 个字节
        payload_hex = ' '.join(byte for byte in payload_bytes)
        #print([timestamp, id_str, dlc, payload_str, payload_bytes, payload_hex])
        payloads.append(payload_hex)

    # 创建 DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'ID': ids,
        'Payload': payloads
    })

    # 保存为 CSV 文件
    data.to_csv(save_path, index=False, header=False)


def can_intrusion_dataset_transform():
    read_path = "dataset/can_intrusion/"
    file_name = "Attack_free_dataset.txt"
    dataset_path = os.path.join(read_path, file_name)
    print(f"read path: {dataset_path}")

    csv_save_file = "attack_free.csv"
    save_path = os.path.join(read_path, csv_save_file)
    print(f"save_path: {save_path}")

    # 读取 TXT 文件
    with open(dataset_path, 'r') as file:
        data_str = file.read()

    # 初始化空列表,用于存储数据
    timestamps = []
    ids = []
    dlcs = []
    payloads = []

    # 按行分割数据
    lines = data_str.strip().split('\n')

    # 遍历每一行,提取数据
    for line in lines:  # 遍历除最后一行之外的所有行,最后一行数据破损
        print(line)

        # 提取时间戳，单位转化为us
        timestamp = int(float(line.split('Timestamp: ')[1].strip().split(' ')[0]) * 1000000)
        #print("timestamp: ", timestamp)

        # 提取 ID
        id_str = line.split('ID: ')[1].split(' ')[0]
        #id_int = int(id_str, 16)

        # 提取 DLC
        dlc = int(line.split('DLC: ')[1].split(' ')[0])

        # 跳过远程帧
        if dlc == 0:
            continue

        # 提取 Payload 并填充为 8 个字节
        #print(line.split(' '))
        #print(line.split('DLC: ')[1].split(' ')[4:])
        payload_str = line.split('DLC: ')[1].split(' ')[4:]
        #payload_bytes = [int(x, 16) for x in payload_str[0:dlc]]
        #payload_bytes = payload_bytes + ["00"] * (8 - len(payload_bytes))  # 填充为 8 个字节
        payload_bytes = payload_str + ["00"] * (8 - len(payload_str))  # 填充为 8 个字节
        payload_hex = ' '.join(byte for byte in payload_bytes)
        #print([timestamp, id_str, dlc, payload_str, payload_bytes, payload_hex])

        timestamps.append(timestamp)
        ids.append(id_str)
        dlcs.append(dlc)
        payloads.append(payload_hex)

    # 创建 DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'ID': ids,
        'Payload': payloads
    })

    # 保存为 CSV 文件
    data.to_csv(save_path, index=False, header=False)


def survival_dataset_transform():
    read_path = "dataset/survival/"
    file_paths = os.listdir(read_path)

    for file_path in file_paths:

        file_names = os.path.join(read_path, file_path)
        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:
            pattern = r'FreeDrivingData_\w+\.txt'

            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                dataset_path = os.path.join(read_path, file_path, file_name)

                print(f"dataset_path: {dataset_path}")

                csv_save_file = "attack_free.csv"
                save_path = os.path.join(read_path, file_path, csv_save_file)
                print(f"save_path: {save_path}")

                # 读取 TXT 文件
                with open(dataset_path, 'r') as file:
                    data_str = file.read()

                # 初始化空列表,用于存储数据
                timestamps = []
                ids = []
                dlcs = []
                payloads = []

                # 按行分割数据
                lines = data_str.strip().split('\n')

                # 遍历每一行,提取数据
                for line in lines:  # 遍历除最后一行之外的所有行,最后一行数据破损
                    print(line)

                    line_value = line.split(',')
                    print(line_value)

                    # 提取时间戳，单位转化为us
                    timestamp = int(float(line_value[0]) * 1000000)

                    # 提取 ID
                    id_str = line_value[1]

                    # 提取 DLC
                    dlc = int(line_value[2])

                    # 提取 Payload 并填充为 8 个字节
                    payload_str = line_value[3:]
                    payload_bytes = None
                    if dlc < 8:
                        payload_bytes = payload_str + ["00"] * (8 - dlc)  # 填充为 8 个字节
                    else:
                        payload_bytes = payload_str
                    payload_hex = ' '.join(byte for byte in payload_bytes)
                    #print([timestamp, id_str, dlc, payload_str, payload_bytes, payload_hex])

                    timestamps.append(timestamp)
                    ids.append(id_str)
                    dlcs.append(dlc)
                    payloads.append(payload_hex)

                # 创建 DataFrame
                data = pd.DataFrame({
                    'Timestamp': timestamps,
                    'ID': ids,
                    'Payload': payloads
                })

                # 保存为 CSV 文件
                data.to_csv(save_path, index=False, header=False)


def vtc2019fall_dataset_transform():
    read_path = "dataset/VTC2019Fall/clean/"
    file_name = "V40_01.txt"
    dataset_path = os.path.join(read_path, file_name)
    print(f"read path: {dataset_path}")

    csv_save_file = "attack_free.csv"
    save_path = os.path.join(read_path, csv_save_file)
    print(f"save_path: {save_path}")

    # 读取 TXT 文件
    with open(dataset_path, 'r') as file:
        data_str = file.read()

    # 初始化空列表,用于存储数据
    timestamps = []
    ids = []
    dlcs = []
    payloads = []

    # 按行分割数据
    lines = data_str.strip().split('\n')

    # 遍历每一行,提取数据
    for line in lines[1:]:  # 该数据集第一行为属性，读取时不需要
        #print(line)

        line_value = line.split(',')
        #print(line_value)

        # 提取时间戳，单位转化为us
        timestamp = int(float(line_value[0]) * 1000000)

        # 提取 ID
        id_str = line_value[1]

        # 提取 DLC
        dlc = int(line_value[2])/8
        # 提取 Payload 并填充为 8 个字节
        payload_str = line_value[3]

        # 将字符串拆分为每两个字符一组
        bytes_list = [payload_str[i:i + 2] for i in range(0, len(payload_str), 2)]

        payload_bytes = None
        if dlc < 8:
            print("dlc < 8")
            payload_bytes = bytes_list + ["00"] * (8 - dlc)  # 填充为 8 个字节
        else:
            payload_bytes = bytes_list
        payload_hex = ' '.join(byte for byte in payload_bytes)
        # print([timestamp, id_str, dlc, payload_str, payload_bytes, payload_hex])

        #print(f"timestamps: {timestamp}, id: {id_str}, dlc: {dlc}, payload: {payload_hex}")
        timestamps.append(timestamp)
        ids.append(id_str)
        dlcs.append(dlc)
        payloads.append(payload_hex)

    # 创建 DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'ID': ids,
        'Payload': payloads
    })

    # 保存为 CSV 文件
    data.to_csv(save_path, index=False, header=False)


def can_train_and_test_dataset_transform():
    read_path = "dataset/can_train_and_test/"
    file_paths = os.listdir(read_path)

    for file_path in file_paths:

        file_names = os.path.join(read_path, file_path)
        file_names = os.listdir(file_names)
        # print(file_names)

        # 遍历每个文件名
        for file_name in file_names:
            pattern = r'attack-free-1.csv'

            # 判断文件名是否匹配统一格式
            if re.match(pattern, file_name):
                dataset_path = os.path.join(read_path, file_path, file_name)

                print(f"dataset_path: {dataset_path}")

                csv_save_file = "attack_free.csv"
                save_path = os.path.join(read_path, file_path, csv_save_file)
                print(f"save_path: {save_path}")

                # 读取文件
                with open(dataset_path, 'r') as file:
                    data_str = file.read()

                # 初始化空列表,用于存储数据
                timestamps = []
                ids = []
                dlcs = []
                payloads = []

                # 按行分割数据
                lines = data_str.strip().split('\n')

                # 遍历每一行,提取数据
                for line in lines[1:]:  # 第一行为属性，不需要
                    #print(line)

                    line_value = line.split(',')
                    #print(line_value)

                    # 提取时间戳，单位转化为us
                    timestamp = int(float(line_value[0]) * 1000000)

                    # 提取 ID
                    id_str = line_value[1]

                    # 提取 Payload 并填充为 8 个字节
                    payload_str = line_value[2]

                    # 将字符串拆分为每两个字符一组
                    bytes_list = [payload_str[i:i + 2] for i in range(0, len(payload_str), 2)]

                    # 计算 DLC
                    dlc = len(bytes_list)
                    payload_bytes = None
                    if dlc < 8:
                        payload_bytes = bytes_list + ["00"] * (8 - dlc)  # 填充为 8 个字节
                    else:
                        payload_bytes = bytes_list
                    payload_hex = ' '.join(byte for byte in payload_bytes)

                    #print(f"timestamps: {timestamp}, id: {id_str}, dlc: {dlc}, payload: {payload_hex}")
                    timestamps.append(timestamp)
                    ids.append(id_str)
                    dlcs.append(dlc)
                    payloads.append(payload_hex)

                # 创建 DataFrame
                data = pd.DataFrame({
                    'Timestamp': timestamps,
                    'ID': ids,
                    'Payload': payloads
                })

                # 保存为 CSV 文件
                data.to_csv(save_path, index=False, header=False)

def main():
    # car_hacking_dataset_transform()        # 将car_hacking正常驾驶数据集从txt转换成csv
    # can_intrusion_dataset_transform()      # 将can_intrusion正常驾驶数据集从txt转换成csv
    # survival_dataset_transform()           # 将survival正常驾驶数据集从txt转换成csv
    # vtc2019fall_dataset_transform()        # 将vtc2019fall正常驾驶数据集从txt转换成csv
    can_train_and_test_dataset_transform()   # 将can_train_and_test正常驾驶数据集转换成统一csv格式

if __name__ == '__main__':
    main()