import math
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from common import *
from my_model import *


# 数据集报文统计
def dataset_statics():
    for file_num, file_path in enumerate(file_paths):

        #print(f"Processing file: {file_path}")

        # 读取数据
        with open(file_path, 'r') as file:
            data_str = file.read()

        # 按行分割数据
        lines = data_str.strip().split('\n')

        frame_count = 0
        frame_ids = []

        first_timestamp = -1
        last_timestamp = -1

        # 遍历每一行,提取数据
        for line in lines:
            # print(line)

            line_value = line.split(',')
            timestamp = int(line_value[0])

            if first_timestamp == -1:
                first_timestamp = timestamp

            last_timestamp = timestamp

            # 提取 ID
            can_id = int(line_value[1], 16)

            # 提取 Payload
            payload_str = line_value[2]
            frame_count += 1

            if can_id not in frame_ids:
                frame_ids.append(can_id)

        during_time = (last_timestamp - first_timestamp)/1000000.0

        fps = frame_count/during_time

        print(f"file_num: {file_num}, Vehicle: {vehicle_names[file_num]}, "
              f"ID count: {len(frame_ids)}, frame count: {frame_count}, fps: {fps}")


# 固定块长度下的取值统计（以字节为单位尝试）
def test_bits_value_range(block_size=8):
    # block_size = 2,4,8,16     当取值位32时，出现内存错误
    for file_num, file_path in enumerate(file_paths):

        # for debug
        #if file_num != 1:
        #    continue

        print(f"Processing file: {file_path}")

        # 读取数据
        with open(file_path, 'r') as file:
            data_str = file.read()

        # 读取数据存到reports列表中
        reports = []
        for line in data_str.strip().split("\n"):
            timestamp, can_id, payload = line.split(",")
            can_id = int(can_id, 16)

            payload_bytes = [x for x in payload.strip().split()]
            # 将负载值转化成64位的数值
            payload_bin_str = ""
            for x in payload_bytes:
                # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                payload_bin_str += bin(int(x, 16))[2:].zfill(8)
                #print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

            # 根据划分的长度拆分负载
            bits_list = [int(payload_bin_str[i:i + block_size], 2) for i in range(0, len(payload_bin_str), block_size)]

            #print(f"payload: {payload}")
            #print(f"payload_bin_str: {payload_bin_str}, len: {len(payload_bin_str)} bits")
            #print(f"bits_list: {bits_list}, len: {len(bits_list)} blocks")

            reports.append((can_id, bits_list))

        # 统计每个字节可能出现的值个数
        result = defaultdict(lambda: [0] * pow(2, block_size))
        for can_id, bits_list in reports:
            for block_pos, bits_value in enumerate(bits_list):
                result[(can_id, block_pos)][bits_value] += 1

        # 输出结果
        for (can_id, block_pos), counts in result.items():
            print(f"ID: 0x{can_id:X}, block_pos: {block_pos}")
            total_count = len([count for count in counts if count > 0])
            print(f"  total number: {total_count}")
            bits_values = []
            for j, count in enumerate(counts):
                if count > 0:
                    bits_values.append((j, count))
            bits_values.sort(key=lambda x: x[1], reverse=True)
            for bits_value, count in bits_values:
                print(f"  bits value {bits_value:0X}: {count}")
            print()

        # summary
        total_ids = len(set(can_id for can_id, _ in reports))
        total_blocks = total_ids * 64/block_size
        bits_count_summary = [0] * (pow(2, block_size)+1)
        for (can_id, block_pos), counts in result.items():
            bits_count = len([count for count in counts if count > 0])
            bits_count_summary[bits_count] += 1

        print(f"total_ids: {total_ids}")
        print(f"total_bits: {total_blocks}")
        #for j, count in enumerate(bits_count_summary):
        #    if count > 0:
        #        print(f"bits possible value number = {j}: {count}")

        #print("bits_distance_value_counts: ", bits_count_summary)

        # 以范围的形式统计字节值的个数
        bits_value_counts_range = [0] * (block_size + 1)
        print("bits_distance_value_counts: ", bits_value_counts_range)

        for i, count in enumerate(bits_count_summary):
            if count > 0:
                pos = math.ceil(math.log2(i))
                bits_value_counts_range[pos] += count
        print("bits_value_counts_range: ", bits_value_counts_range)

        # 计算每个数字出现的百分比
        total_counts = np.sum(bits_value_counts_range)
        percentages = [round(item/total_counts * 100, 2) for item in bits_value_counts_range]   # round(number, 2) 保留两位小数
        print("percentages: ", percentages)

        # 绘制条形图，显示数据分布
        index = np.arange(len(bits_value_counts_range))
        y_data = bits_value_counts_range
        #plt.clf()
        plt.figure(figsize=(8, 6))
        plt.bar(index, y_data, color='steelblue', edgecolor='black')
        for i in range(len(index)):
            plt.text(i, y_data[i], f"{y_data[i]}\n({percentages[i]}%)", ha='center', va='bottom')
        plt.xticks(index)
        plt.xlabel("Encoding length of dictionary")
        plt.ylabel("Frequency")
        # 调整 y 轴范围
        max_height = max(y_data)
        plt.ylim(0, max_height * 1.1)

        # 添加横跨的括号注释
        index1 = 5
        index2 = 8
        h = max_height/3  # maximum of the involved bar heights
        bx = [index1 - 0.5, index1 - 0.5, index2 + 0.5, index2 + 0.5]
        by = [h * (1 + 0.05), h * (1 + 0.1), h * (1 + 0.1), h * (1 + 0.05)]
        plt.plot(bx, by, "r-", linewidth=1)
        plt.text((index1 + index2) / 2, h * 1.2, f'{sum(percentages[index1:index2 + 1]):.2f} %', ha='center', va='bottom', color="r")

        bool_figure_save = True
        if bool_figure_save:
            figure_name = f"{file_num}_{vehicle_names[file_num]}"
            folder_path = "evaluation_result/byte_value_count"
            savefig_path = os.path.join(folder_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()


# 固定块长度下的连续2帧差值的取值统计（以字节为单位尝试）
def test_delta_value_range(block_size=8):
    # block_size = 2,4,8,16     当取值位32时，出现内存错误
    for file_num, file_path in enumerate(file_paths):

        # for debug
        #if file_num != 1:
        #    continue

        print(f"Processing file: {file_path}")

        # 读取数据
        with open(file_path, 'r') as file:
            data_str = file.read()

        # 读取数据存到reports列表中
        reports = []
        # 存储之前的值
        previous_cache = {}

        upper_bound = pow(2, block_size)

        for line in data_str.strip().split("\n"):
            timestamp, can_id, payload = line.split(",")
            can_id = int(can_id, 16)

            payload_bytes = [x for x in payload.strip().split()]
            # 将负载值转化成64位的数值
            payload_bin_str = ""
            for x in payload_bytes:
                # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                payload_bin_str += bin(int(x, 16))[2:].zfill(8)
                #print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

            # 根据划分的长度拆分负载
            bits_list = [int(payload_bin_str[i:i + block_size], 2) for i in range(0, len(payload_bin_str), block_size)]

            #print(f"payload: {payload}")
            #print(f"payload_bin_str: {payload_bin_str}, len: {len(payload_bin_str)} bits")
            #print(f"bits_list: {bits_list}, len: {len(bits_list)} blocks")

            if can_id in previous_cache:
                previous_bits_list = previous_cache[can_id]
                delta_list = []
                for x, y in zip(previous_bits_list, bits_list):
                    diff = y - x
                    delta_list.append((diff + upper_bound) % upper_bound if diff < 0 else diff % upper_bound)

                reports.append((can_id, delta_list))

            # 更新缓存
            previous_cache[can_id] = bits_list

        # 统计每个字节可能出现的值个数
        result = defaultdict(lambda: [0] * pow(2, block_size))
        for can_id, bits_list in reports:
            for block_pos, bits_value in enumerate(bits_list):
                result[(can_id, block_pos)][bits_value] += 1

        # 输出结果
        for (can_id, block_pos), counts in result.items():
            print(f"ID: 0x{can_id:X}, block_pos: {block_pos}")
            total_count = len([count for count in counts if count > 0])
            print(f"  total number: {total_count}")
            delta_values = []
            for j, count in enumerate(counts):
                if count > 0:
                    delta_values.append((j, count))
            delta_values.sort(key=lambda x: x[1], reverse=True)
            for delta_value, count in delta_values:
                print(f"  delta value {delta_value:0X}: {count}")
            print()

        # summary
        total_ids = len(set(can_id for can_id, _ in reports))
        total_blocks = total_ids * 64/block_size
        delta_count_summary = [0] * (pow(2, block_size)+1)
        for (can_id, block_pos), counts in result.items():
            delta_count = len([count for count in counts if count > 0])
            delta_count_summary[delta_count] += 1

        print(f"total_ids: {total_ids}")
        print(f"total_bits: {total_blocks}")
        #for j, count in enumerate(delta_count_summary):
        #    if count > 0:
        #        print(f"delta possible value number = {j}: {count}")

        #print("delta_count_summary: ", delta_count_summary)

        # 以范围的形式统计字节值的个数
        delta_value_counts_range = [0] * (block_size + 1)
        print("initial delta_value_counts_range: ", delta_value_counts_range)

        for i, count in enumerate(delta_count_summary):
            if count > 0:
                pos = math.ceil(math.log2(i))
                delta_value_counts_range[pos] += count
        print("delta_value_counts_range: ", delta_value_counts_range)

        # 计算每个数字出现的百分比
        total_counts = np.sum(delta_value_counts_range)
        percentages = [round(item/total_counts * 100, 2) for item in delta_value_counts_range]   # round(number, 2) 保留两位小数
        print("percentages: ", percentages)

        # 绘制条形图，显示数据分布
        index = np.arange(len(delta_value_counts_range))
        y_data = delta_value_counts_range
        #plt.clf()
        plt.figure(figsize=(8, 6))
        plt.bar(index, y_data, color='steelblue', edgecolor='black')
        for i in range(len(index)):
            plt.text(i, y_data[i], f"{y_data[i]}\n({percentages[i]}%)", ha='center', va='bottom')
        plt.xticks(index)
        plt.xlabel("Encoding length of dictionary")
        plt.ylabel("Frequency")
        # 调整 y 轴范围
        max_height = max(y_data)
        plt.ylim(0, max_height * 1.1)

        # 添加横跨的括号注释
        index1 = 5
        index2 = 8
        h = max_height/3  # maximum of the involved bar heights
        bx = [index1 - 0.5, index1 - 0.5, index2 + 0.5, index2 + 0.5]
        by = [h * (1 + 0.05), h * (1 + 0.1), h * (1 + 0.1), h * (1 + 0.05)]
        plt.plot(bx, by, "r-", linewidth=1)
        plt.text((index1 + index2) / 2, h * 1.2, f'{sum(percentages[index1:index2 + 1]):.2f} %', ha='center', va='bottom', color="r")

        bool_figure_save = True
        if bool_figure_save:
            figure_name = f"{file_num}_{vehicle_names[file_num]}"
            folder_path = "evaluation_result/delta_value_count"
            savefig_path = os.path.join(folder_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()

# 不同块大小的性能比较
def proposed_block_size_selection():
    """
    设置不同的bit_length [2, 4, 8, 16]，比较压缩性能
    :return:
    """

    # story the performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        #if file_num != 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表  bit_lengths = [2, 4, 8, 16]
        models = [
            TBDR(block_size=2),
            TBDR(block_size=4),
            TBDR(block_size=8),
            TBDR(block_size=16),
        ]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):
            # 创建模型实例
            model_name = model.__class__.__name__ + f"-{model_num:03d}"

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            # 根据数据获取静态字典
            model.dictionary_building(data_str)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                # print(f"can_id: {can_id}, original_message: {byte_data}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                # print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # update the dictionary
                model.dictionary_update(can_id, byte_data)

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)
            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 2))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))            # [9] memory_footprint
            performance_results.append(current_result)

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/proposed_block_size_selection.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 不同训练比例的性能比较
def proposed_train_ratio_comparison():

    # story the model performance
    performance_results = []

    model_quantity = 10    # [0.1~1.0]
    for file_num, file_path in enumerate(file_paths):

        #if file_num != 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表

        models = [TBDR(read_ratio=i / 10) for i in range(1, model_quantity + 1, 1)]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):
            # 创建模型实例
            model_name = model.__class__.__name__ + f"-{model_num:03d}"
            #print(f"Applying model: {model_name}")

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            # 根据数据获取静态字典
            model.dictionary_building(data_str)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                #print(f"can_id: {can_id}, original_message: {byte_data}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                #print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # update the dictionary
                model.dictionary_update(can_id, byte_data)

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)
            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 2))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))            # [9] memory_footprint
            performance_results.append(current_result)

    # 折线图显示不同训练比例的平均压缩率
    total_average_compression_level = []
    for item in performance_results:
        total_average_compression_level.append(item[3])
    # 以车分组，每组10个，包含 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_ratios = [i / 10 for i in range(1, model_quantity + 1, 1)]
    group_average_compression_level = [total_average_compression_level[i:i + model_quantity] for i in range(0, len(total_average_compression_level), model_quantity)]

    # 20 vehicles, each color for one vehicle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf', '#1a55FF', '#FF69B4', '#BDB76B', '#FFA500', '#800000', '#008080', '#000080', '#808000',
              '#FF00FF', '#C0C0C0']  # Example, add more if needed

    for vehicle_num, compression_levels in enumerate(group_average_compression_level):
        plt.plot(train_ratios, compression_levels, label=f"{vehicle_num+1}", color=colors[vehicle_num])

    plt.grid()
    plt.xticks(train_ratios)
    plt.xlabel("Train ratios")
    plt.ylabel("Mean compression level (bits)")
    plt.legend(ncol=2, loc='best')
    figure_name = "proposed_train_ratio_comparison"
    savefig_path = "evaluation_result/compression_effect/" + figure_name
    plt.savefig(savefig_path, dpi=300)
    plt.close()

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/proposed_train_ratio_comparison.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 优化前后的性能比较
def proposed_optimization_comparison():

    # story the model performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        #if file_num != 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            TBDR(),
            #ITBDR(silence_threshold=1),
            #ITBDR(silence_threshold=2),
            #ITBDR(silence_threshold=3),
            #ITBDR(silence_threshold=4),
            #ITBDR(silence_threshold=5),
            ITBDR(silence_threshold=6),
            #ITBDR(silence_threshold=7),
            #ITBDR(silence_threshold=8),      # TBDR(),
        ]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):
            # 创建模型实例
            model_name = model.__class__.__name__

            if model_name == "ITBDR":
                model_name += f"({model.silence_threshold})"

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            # 根据数据获取静态字典
            model.dictionary_building(data_str)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                # print(f"can_id: {can_id}, original_message: {byte_data}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                # print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # update the dictionary
                model.dictionary_update(can_id, byte_data)

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)
            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 2))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))            # [9] memory_footprint
            performance_results.append(current_result)

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/proposed_optimization_comparison.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 压缩性能分布可视化
def proposed_compression_distribution():

    # story the model performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        #if file_num != 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            #ITBDR(),
            IVDCTBDR(),
        ]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):
            # 创建模型实例
            model_name = model.__class__.__name__

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            # 根据数据获取静态字典
            model.dictionary_building(data_str)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                #print(f"can_id: {can_id}, original_message: {byte_data}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                #print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # update the dictionary
                model.dictionary_update(can_id, byte_data)

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)

            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 4))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))                        # [9] memory_footprint
            performance_results.append(current_result)

            plot_flag = True
            if plot_flag:
                # plt.clf()
                plt.figure(figsize=(8, 6))
                plt.bar(index, y_data, color='steelblue', edgecolor='black')

                # 画一条竖直的线显示最低压缩
                plt.axvline(x=first_positive_index, color="r", linewidth=0.5)
                plt.text(first_positive_index + 1, max(y_data) * 2 / 3, f"min={first_positive_index}", color="r", ha='left',
                         va='center')
                # 画一条竖直的线显示最高压缩
                plt.axvline(x=last_positive_index, color="g", linewidth=0.5)
                plt.text(last_positive_index + 1, max(y_data) * 2 / 3, f"max={last_positive_index}", color="g", ha='left',
                         va='center')

                # 数据添加注释
                # for i in index:
                #    if y_data[i]>0:
                #        plt.text(i, y_data[i], f"{y_data[i]}\n({percentages[i]}%)", ha='center', va='bottom')
                # plt.xticks(index)
                plt.xlabel("Compression level (bits)")
                plt.ylabel("Frequency")

                # 调整 y 轴范围,确保注释不超出边界
                # max_height = max(y_data)
                # plt.ylim(0, max_height * 1.1)

                # plt.pie(percentages)
                # plt.title("Data Distribution")

                bool_figure_save = True

                if bool_figure_save:
                    figure_name = f"{file_num}_{vehicle_names[file_num]}"
                    folder_path = "evaluation_result/compression_effect/compression_distribution/"
                    savefig_path = os.path.join(folder_path, figure_name)
                    plt.savefig(savefig_path, dpi=300)
                    plt.close()
                else:
                    plt.show()

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/compression_distribution.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 不同方法性能比较
def methods_comparison():

    # story the model performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        #if file_num == 0 or file_num > 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            Baseline(),
            CASmap(),
            SFDC(read_ratio=0.5),
            TBDR(read_ratio=0.5),
            ITBDR(read_ratio=0.5, silence_threshold=5),
            TDBDR(read_ratio=0.5),
            ITDBDR(read_ratio=0.5, silence_threshold=5),
            VDCTBDR(read_ratio=0.5),
            IVDCTBDR(read_ratio=0.5, silence_threshold=5),
        ]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):

            # 创建模型实例
            model_name = model.__class__.__name__

            print(f"Applying model: {model_name}")

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            if model_name == "Baseline" or model_name == "CASmap":
                nothing = 0
            elif model_name == "SFDC":
                model.mapping_matrix_building(data_str)
            else:
                # 根据数据获取静态字典
                model.dictionary_building(data_str)


            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            read_count = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                #print(f"can_id: {can_id}, original_message: {byte_data}")
                #print(f"encoding_message_bin: {message_str}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                #print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # update the dictionary
                if model_name == "Baseline" or model_name == "CASmap":
                    model.message_cache_update(can_id, byte_data)
                elif model_name == "SFDC":
                    nothing = 0
                else:
                    model.dictionary_update(can_id, byte_data)

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

                # for debug
                read_count += 1
                #if read_count > 50:
                #    break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)
            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 2))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))            # [9] memory_footprint
            performance_results.append(current_result)

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/methods_comparison.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    #print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        #print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    #print(performance_results)
    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 不同训练比例的性能比较
def sfdc_train_ratio_comparison():

    # story the model performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        #if file_num != 3:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [SFDC(read_ratio=i/10) for i in range(1, 11, 1)]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):
            # 创建模型实例
            model_name = model.__class__.__name__ + f"-{model_num:03d}"
            #print(f"Applying model: {model_name}")

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            # 根据数据获取静态字典
            model.mapping_matrix_building(data_str)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 计算可能的压缩值
            y_bit_compression = []
            # 时间统计
            encoding_times_list = []
            decoding_times_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 遍历每一行,提取数据
            for line in lines:
                # print(line)

                line_value = line.split(',')
                timestamp = int(line_value[0])

                # 提取 ID
                can_id = int(line_value[1], 16)

                # 提取 Payload
                payload_str = line_value[2]
                # 负载按照空格分割
                byte_data_str = payload_str.strip().split(' ')

                byte_data = [x for x in byte_data_str]

                # encoding the message
                start_time = time.time()
                message_str = model.message_encoding(can_id, byte_data)
                end_time = time.time()
                encoding_elapsed_time = end_time - start_time
                encoding_times_list.append(encoding_elapsed_time)

                # 超出8字节长度，视为无法压缩
                if len(message_str) > 64:
                    total_free_bit_len = 0
                    y_bit_compression.append(total_free_bit_len)
                else:
                    total_free_bit_len = 64 - len(message_str)
                    y_bit_compression.append(total_free_bit_len)

                # 将二进制字符串划分为8位一组
                bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                # 将每个8位二进制字符串转换为16进制字符串
                hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                #print(f"can_id: {can_id}, original_message: {byte_data}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # decoding the message
                start_time = time.time()
                m_payload = model.message_decoding(can_id, message_str)
                end_time = time.time()
                decoding_elapsed_time = end_time - start_time
                decoding_times_list.append(decoding_elapsed_time)
                #print(f"decoding_payload: {m_payload}, decoding_elapsed_time: {decoding_elapsed_time / 1e-6} us")

                # 判断是否编解码是否错误
                for i in range(8):
                    if int(m_payload[i], 16) != int(byte_data[i], 16):
                        error_counts += 1
                        break

            bit_compression_value_counts = [0] * 65  # [0, 64]
            # print("bit_compression_value_counts: ", bit_compression_value_counts)

            for item in y_bit_compression:
                bit_compression_value_counts[item] += 1

            #print("bit_compression_value_counts: ", bit_compression_value_counts)

            # 计算每个数字出现的百分比
            total_counts = np.sum(bit_compression_value_counts)
            percentages = [item / total_counts * 100 for item in bit_compression_value_counts]

            # 计算压缩等级小于1字节的比例
            low_one_byte_ratio = np.sum(percentages[:9])
            #print("percentages: ", percentages)

            # 绘制条形图，显示数据分布
            index = np.arange(len(bit_compression_value_counts))
            y_data = bit_compression_value_counts
            first_positive_index = np.where(np.array(y_data) > 0)[0][0]  # 获取最低压缩值
            #print(f"first_positive_index: {first_positive_index}")

            mean_compression_level = np.mean(y_bit_compression)  # 获取平均压缩值

            last_positive_index = np.where(np.array(y_data) > 0)[0][-1]  # 获取最高压缩值
            #print(f"last_positive_index: {last_positive_index}")

            average_encoding_time = np.mean(encoding_times_list)
            average_decoding_time = np.mean(decoding_times_list)
            error_ratio = error_counts / total_counts * 100
            memory_footprint = get_total_size(model)

            current_result = []
            current_result.append(file_num)                                # [0] vehicle number
            current_result.append(model_name)                              # [1] bit_length
            current_result.append(first_positive_index)                    # [2] min_compression_level
            current_result.append(round(mean_compression_level, 2))        # [3] mean_compression_level
            current_result.append(last_positive_index)                     # [4] max_compression_level
            current_result.append(round(low_one_byte_ratio, 2))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 4))                   # [8] error_ratio
            current_result.append(round(memory_footprint/1000))                        # [9] memory_footprint
            performance_results.append(current_result)

    # 折线图显示不同训练比例的平均压缩率
    total_average_compression_level = []
    for item in performance_results:
        total_average_compression_level.append(item[3])
    # 以车分组，每组10个，包含 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    group_average_compression_level = [total_average_compression_level[i:i + 10] for i in range(0, len(total_average_compression_level), 10)]
    for vehicle_num, compression_levels in enumerate(group_average_compression_level):
        plt.plot(train_ratios, compression_levels, label=f"{vehicle_num:03d}")
    plt.grid()
    plt.xticks(train_ratios)
    plt.xlabel("Train ratios")
    plt.ylabel("Mean compression level (bits)")
    plt.legend()
    figure_name = "sfdc_train_ratio_comparison"
    savefig_path = "evaluation_result/compression_effect/" + figure_name
    plt.savefig(savefig_path, dpi=300)
    plt.close()

    save_res_flag = True
    save_res_path = "evaluation_result/compression_effect/sfdc_train_ratio_comparison.txt"
    res_file = None
    if save_res_flag:
        res_file = open(save_res_path, "w")

    # print the comparison results
    str_size = []
    for i in range(len(performance_names)):
        str_size.append(len(performance_names[i]) + 5)
    # print("str_size", str_size)
    output_head = ""
    for i, mpn in enumerate(performance_names):
        # print(f"{mpn:>{str_size[i]}}")
        output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()


# 压缩前后的总线负载变化
def bus_load_comparison():

    for file_num, file_path in enumerate(file_paths):

        #if file_num != 0:
        #    continue
        print(f"Processing file: {file_path}")

        # 创建模型
        #model = ITBDR()
        model = IVDCTBDR()

        # Read data
        with open(file_path, 'r') as file:
            data_str = file.read()

        # 模型字典训练
        model.dictionary_building(data_str)

        # Split data into lines
        lines = data_str.strip().split('\n')

        frame_count = 0

        label_name = ["Original", "Compressed + 0-bit MAC", "Compressed + 4-bit MAC",
                      "Compressed + 8-bit MAC", "Compressed + 12-bit MAC", "Compressed + 16-bit MAC"]
        mac_length = [0, 0, 4, 8, 12, 16]

        busload_per_second = []                        # 存储原始数据的负载
        for i in range(len(label_name)):
            busload_per_second.append({})

        # Process each line to extract data
        for line in lines:
            line_value = line.split(',')
            timestamp = int(line_value[0])             # in microseconds
            can_id = int(line_value[1], 16)            # extracting CAN ID
            payload_str = line_value[2]                # Payload string

            # Convert timestamp to seconds
            second = timestamp // 1_000_000

            # 提取 Payload
            byte_data_str = payload_str.strip().split(' ')
            byte_data = [x for x in byte_data_str]

            # 字节数
            byte_num = len(byte_data)

            # encoding the message and get block length
            message_str, compression_marker_str, block_length = model.message_encoding_with_detailed_output(can_id, byte_data)

            for i in range(len(mac_length)):
                if i == 0:
                    frame_size = 8 * byte_num + 64
                else:
                    if mac_length[i] != 0:
                        mac_str = ''.join(['0' for _ in range(mac_length[i])])
                        message_str += mac_str
                        #print("mac_length[i]: ", mac_length[i])
                        #print("mac_str: ", mac_str)

                    # 将二进制字符串划分为8位一组
                    bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]

                    # 将每个8位二进制字符串转换为16进制字符串
                    hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]

                    frame_size = 0
                    if len(hex_strings) > 8:
                        # 超过 64 bits
                        # 获取前四个字节和后四个字节压缩后的长度
                        mid_pos = int(len(block_length)/2)
                        payload_size1 = sum(block_length[:mid_pos])
                        payload_size2 = sum(block_length[mid_pos:])

                        #print("hex_strings: ", hex_strings)
                        #print("block_length: ", block_length)
                        #print(f"payload_size1: {payload_size1}, payload_size2: {payload_size2}")

                        # compression_marker_str, 1-bit marker
                        # 计算报文负载大小（比特）
                        payload_size1 = len(compression_marker_str) + 1 + payload_size1 + mac_length[i]
                        payload_size2 = len(compression_marker_str) + 1 + payload_size2 + mac_length[i]

                        # 计算报文负载大小（字节）
                        byte_size1 = math.ceil(payload_size1/8)
                        byte_size2 = math.ceil(payload_size2/8)
                        #print(f"payload_size1: {payload_size1}, payload_size1/8: {payload_size1/8}, byte_size1: {byte_size1}")

                        # 第一帧大小
                        frame_size = 8 * byte_size1 + 64 + math.floor((54 + 8 * byte_size1 - 1) / 4)
                        # 第二帧大小
                        frame_size += 8 * byte_size2 + 64 + math.floor((54 + 8 * byte_size2 - 1) / 4)
                        # 3比特 IFS(Interframe spacing)
                        frame_size += 3

                    else:
                        frame_size = 8 * len(hex_strings) + 64 + math.floor((54 + 8 * len(hex_strings) - 1) / 4)

                # Aggregate payload size by second
                if second not in busload_per_second[i]:
                    busload_per_second[i][second] = 0
                busload_per_second[i][second] += frame_size

            # 模型字典更新
            model.dictionary_update(can_id, byte_data)
            frame_count += 1

        # Determine the range of seconds to keep (full seconds)
        seconds = sorted(busload_per_second[0].keys())
        #print("seconds:", seconds)

        # Filter out head and tail seconds that do not have a full second of data
        full_seconds = seconds[1:len(seconds)-1]

        # Normalize time to start from 0
        normalized_seconds = [sec - full_seconds[0] for sec in full_seconds]

        # Plotting
        plt.figure(figsize=(8, 6))
        for i in range(len(label_name)):

            # Prepare the payload sizes for full seconds(kbit)
            filtered_payload_sizes = [busload_per_second[i][sec] for sec in full_seconds]
            # 以500Kbit率为基准计算负载率
            y = np.array(filtered_payload_sizes)/500000*100
            plt.plot(normalized_seconds, y, label=label_name[i])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Bus load (%)')
        #plt.ylim([-0.5, 500])
        #if file_num == 7 or file_num == 1:
        #    plt.ylim([-0.5, 150.5])
        #else:
        plt.ylim([-5, 105])
        #plt.title(f'Bus load per Second for {vehicle_names[file_num]}')
        plt.legend()
        plt.grid()

        bool_figure_save = True

        if bool_figure_save:
            figure_name = f"{file_num}_{vehicle_names[file_num]}"
            folder_path = "evaluation_result/bus_load/"
            savefig_path = os.path.join(folder_path, figure_name)
            plt.savefig(savefig_path, dpi=300)
            plt.close()
        else:
            plt.show()


def hmac_test():
    # 示例使用
    key = b'0123456789abcdef'
    can_message = b'\x01\x23\x45\x67\x89\xAB\xCD\xEF'

    array_time = []
    for i in range(1000):
        t1 = time.perf_counter()                       # 两次time.perf_counter()的差值，返回的单位为seconds
        auth_tag = hmac_can_message(key, can_message)
        t2 = time.perf_counter()
        #print(f"HMAC Authentication Tag: {auth_tag.hex()}")
        array_time.append(t2 - t1)
    average_time = np.mean(array_time)
    print(f"HMAC average calculation time: {average_time * 1000:.6f} ms")


def balance_visualization():

    x = [100, 25, 0, 12.5, 0]
    y = [32, 16, 5, 8, 15]
    label_name = ["Method 1", "Method 2", "Method 3", "Method 4", "Proposed"]
    marks = ["o", "^", "d", "+", "s"]     # "x" "*" "p"
    colors = ["blue", "black", "magenta", "orange", "red"]

    #color_names = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'purple', 'brown', 'orange',
    #               'pink']

    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    for i in range(len(x)):
        plt.scatter(x[i], y[i], marker=marks[i], label=label_name[i], color= colors[i])
        #plt.annotate(label_name[i], (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='left')

    plt.scatter(0, 32, marker="*", color="green")
    #plt.annotate(0, 32, str= "Expected position (0, 32)")
    #plt.annotate('Expected position (0, 32)', xy=(0, 32), xytext=(20, 32), textcoords='offset points',
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.arrow(20, 27, -20, 5, head_width=0.5, head_length=0.5, fc='lightblue', ec='black')  # x,y dx,dy

    plt.annotate('Ideal position (0, 32)', xy=(2, 32), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', color="green",
                 fontsize=16, arrowprops=dict(color='green', arrowstyle='->',  connectionstyle='arc3,rad=-0.2'))

    plt.xlabel('Equivalent increase in bus load (%)')
    plt.ylabel('Security level (bits)')
    #plt.legend(loc="best")
    plt.legend(loc="right")
    #plt.tight_layout()
    plt.xlim(-1.05, 105.05)
    plt.ylim(-0.05, 33.05)
    plt.savefig('evaluation_result/security_balance.png', dpi=300)
    plt.show()
    plt.close()  # Close the plot to free up resources

def main():
    # dataset_statics()                                                 # 数据集统计
    # test_bits_value_range()                                            # 固定块长度下的取值统计，长度可取[2,4,8,16]，默认为8bits=1byte
    # test_delta_value_range()
    # proposed_block_size_selection()                                    # 不同块大小的性能比较
    # proposed_train_ratio_comparison()                                  # 不同训练比例的性能比较
    # proposed_optimization_comparison()                                  # 不同优化阈值下的性能比较
    # proposed_compression_distribution()                                  # 压缩性能分布可视化
    # methods_comparison()                                               # 不同方法性能比较
    # sfdc_train_ratio_comparison()                                      # sfdc方法在不同比例训练数据下的效果
    # bus_load_comparison()                                                 # 压缩前后的总线负载变化
    # hmac_test()                                                          # 计算HMAC的耗时
    balance_visualization()                                            # 安全性负载平衡可视化

if __name__ == '__main__':
    main()