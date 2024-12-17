import os
import time

import can
from common import *
from my_model import *
import numpy as np


def can0_send():

    os.system('sudo ip link set can0 type can bitrate 500000')
    os.system('sudo ifconfig can0 up')
    can0 = can.interface.Bus(channel='can0', bustype='socketcan')     # socketcan_native

    # story the model performance
    performance_results = []

    for file_num, file_path in enumerate(file_paths):

        if file_num != 1:
            continue
        print(f"Processing file: {file_path}")

        # 创建模型并进行训练
        # 定义要尝试的模型列表
        models = [
            # ITBDR(),
            IVDCTBDR(),
        ]

        # 循环遍历每个模型
        for model_num, model in enumerate(models):

            # 创建模型实例
            model_name = model.__class__.__name__

            print(f"Applying model: {model_name}")

            # 读取数据
            with open(file_path, 'r') as file:
                data_str = file.read()

            t0 = time.time()
            model.dictionary_building(data_str)
            dictionary_building_time = round(time.time() - t0, 2)

            # 再次读取数据,进行压缩测试
            # 按行分割数据
            lines = data_str.strip().split('\n')

            # 时间统计
            step1_times_list = []
            step2_times_list = []
            step3_times_list = []
            step3_2_times_list = []
            updating_times_list = []
            latency_list = []
            latency2_list = []

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

                print(f"read_count: {read_count}, can_id: {can_id}, original_message: {byte_data}")

                # start the algorithm process
                t1 = time.time()
                # encoding the message and get block length
                compression_marker_str, encoding_message_str,  block_length = \
                    model.message_encoding_online(can_id, byte_data)
                step1_times_list.append(time.time() - t1)

                t2 = time.time()
                expected_len = len(compression_marker_str) + len(encoding_message_str) + minimum_mac_len
                additional_mode = False
                if expected_len > 64:
                    additional_mode = True
                step2_times_list.append(time.time() - t2)

                if additional_mode:

                    t3 = time.time()

                    # 超过 64 bits
                    # 获取前四个字节和后四个字节压缩后的长度
                    mid_pos = int(len(block_length) / 2)
                    payload_size1 = sum(block_length[:mid_pos])
                    payload_size2 = sum(block_length[mid_pos:])
                    #print(f"payload_size1: {payload_size1}, payload_size2: {payload_size2}")

                    # incomplete frame, part one
                    content1 = ''.join(byte_data[:len(byte_data)//2])
                    tag1 = hmac_can_message(secret_key, bytes.fromhex(content1))
                    number_of_bytes = math.ceil((len(compression_marker_str) + 1 + payload_size1 + minimum_mac_len)/8)
                    cc_mac_len = number_of_bytes*8 - (len(compression_marker_str) + 1 + payload_size1)
                    mac_str1 = bin(int(tag1, 16))[2:2+cc_mac_len]
                    #print(f"content1: {content1}, number_of_bytes1: {number_of_bytes}, mac_str1: {mac_str1}")

                    payload1 = compression_marker_str + '0' + encoding_message_str[:payload_size1] + mac_str1
                    # 将二进制字符串划分为8位一组
                    bytes_list1 = [payload1[i:i + 8] for i in range(0, len(payload1), 8)]
                    # 将每个8位二进制字符串转换为16进制字符串
                    hex_strings1 = [int(byte, 2) for byte in bytes_list1]
                    #print(f"encoding_message1: {hex_strings1}")
                    msg1 = can.Message(is_extended_id=True, arbitration_id=can_id, data=hex_strings1)

                    # incomplete frame, part two
                    content2 = ''.join(byte_data[len(byte_data)//2:])
                    tag2 = hmac_can_message(secret_key, bytes.fromhex(content2))
                    number_of_bytes = math.ceil((len(compression_marker_str) + 1 + payload_size2 + minimum_mac_len)/8)
                    cc_mac_len = number_of_bytes*8 - (len(compression_marker_str) + 1 + payload_size2)
                    mac_str2 = bin(int(tag2, 16))[2:2+cc_mac_len]
                    #print(f"content2: {content2}, number_of_bytes2: {number_of_bytes}, mac_str2: {mac_str2}")
                    payload2 = compression_marker_str + '1' + encoding_message_str[payload_size1:] + mac_str2
                    # 将二进制字符串划分为8位一组
                    bytes_list2 = [payload2[i:i + 8] for i in range(0, len(payload2), 8)]
                    # 将每个8位二进制字符串转换为16进制字符串
                    hex_strings2 = [int(byte, 2) for byte in bytes_list2]
                    #print(f"encoding_message2: {hex_strings2}")
                    msg2 = can.Message(is_extended_id=True, arbitration_id=can_id, data=hex_strings2)

                    step3_2_times_list.append(time.time() - t3)

                    can0.send(msg1)
                    can0.send(msg2)

                else:
                    t3 = time.time()

                    # complete frame
                    payload_size = sum(block_length)

                    tag = hmac_can_message(secret_key, bytes.fromhex(''.join(byte_data_str)))
                    number_of_bytes = math.ceil((len(compression_marker_str) + payload_size + minimum_mac_len)/8)
                    cc_mac_len = number_of_bytes*8 - (len(compression_marker_str) + payload_size)
                    mac_str = bin(int(tag, 16))[2:2+cc_mac_len]
                    #print(f"mac_str: {mac_str}")
                    payload = compression_marker_str + encoding_message_str + mac_str

                    # 将二进制字符串划分为8位一组
                    bytes_list = [payload[i:i + 8] for i in range(0, len(payload), 8)]
                    # 将每个8位二进制字符串转换为16进制字符串
                    hex_strings = [int(byte, 2) for byte in bytes_list]
                    #print(f"encoding_message_bin: {message_str}, encoding_message: {hex_strings}")

                    msg = can.Message(is_extended_id=True, arbitration_id=can_id, data=hex_strings)

                    step3_times_list.append(time.time() - t3)

                    can0.send(msg)


                #print(f"encoding_elapsed_time: {round(encoding_elapsed_time / 1e-6, 2)} us ")

                # 将二进制字符串划分为8位一组
                # bytes_list = [message_str[i:i + 8] for i in range(0, len(message_str), 8)]
                # 将每个8位二进制字符串转换为16进制字符串
                # hex_strings = [hex(int(byte, 2))[2:].zfill(2) for byte in bytes_list]
                # print(f"can_id: {can_id}, original_message: {byte_data}")
                # print(f"encoding_message_bin: {message_str}, encoding_message: {hex_strings}, total_free_bit_len: {total_free_bit_len}")

                # update the dictionary
                t4 = time.time()
                model.dictionary_update(can_id, byte_data)
                updating_times_list.append(time.time() - t4)

                latency = time.time() - t1
                if additional_mode:
                    latency2_list.append(latency)
                else:
                    latency_list.append(latency)

                # for debug
                read_count += 1
                #time.sleep(5)
                time.sleep(0.01)
                #if read_count > 50:
                if read_count > 5000:
                #if not complete_frame_flag:
                    break

            print(f"read_count: {read_count}")

            mean_step1_time = round(np.mean(step1_times_list) / 1e-6, 2)
            max_step1_time = round(np.max(step1_times_list) / 1e-6, 2)
            min_step1_time = round(np.min(step1_times_list) / 1e-6, 2)

            mean_step2_time = round(np.mean(step2_times_list) / 1e-6, 2)
            max_step2_time = round(np.max(step2_times_list) / 1e-6, 2)
            min_step2_time = round(np.min(step2_times_list) / 1e-6, 2)

            mean_step3_time = round(np.mean(step3_times_list) / 1e-6, 2)
            max_step3_time = round(np.max(step3_times_list) / 1e-6, 2)
            min_step3_time = round(np.min(step3_times_list) / 1e-6, 2)

            mean_step3_2_time = round(np.mean(step3_2_times_list) / 1e-6, 2)
            max_step3_2_time = round(np.max(step3_2_times_list) / 1e-6, 2)
            min_step3_2_time = round(np.min(step3_2_times_list) / 1e-6, 2)

            mean_updating_time = round(np.mean(updating_times_list) / 1e-6, 2)
            max_updating_time = round(np.max(updating_times_list) / 1e-6, 2)
            min_updating_time = round(np.min(updating_times_list) / 1e-6, 2)

            mean_latency = round(np.mean(latency_list) / 1e-6, 2)
            max_latency = round(np.max(latency_list) / 1e-6, 2)
            min_latency = round(np.min(latency_list) / 1e-6, 2)

            mean_latency2 = round(np.mean(latency2_list) / 1e-6, 2)
            max_latency2 = round(np.max(latency2_list) / 1e-6, 2)
            min_latency2 = round(np.min(latency2_list) / 1e-6, 2)

            print(f"building_time(s), mean: {dictionary_building_time}")
            print(f"step1_time(us), mean: {mean_step1_time}, max: {max_step1_time}, min: {min_step1_time}")
            print(f"step2_time(us), mean: {mean_step2_time}, max: {max_step2_time}, min: {min_step2_time}")
            print(f"step3_time(us), mean: {mean_step3_time}, max: {max_step3_time}, min: {min_step3_time}")
            print(f"step3_2_time(us), mean: {mean_step3_2_time}, max: {max_step3_2_time}, min: {min_step3_2_time}")
            print(f"updating_time(us), mean: {mean_updating_time}, max: {max_updating_time}, min: {min_updating_time}")
            print(f"latency(us), mean: {mean_latency}, max: {max_latency}, min: {min_latency}")
            print(f"latency2(us), mean: {mean_latency2}, max: {max_latency2}, min: {min_latency2}")
    '''
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
            current_result.append(round(low_one_byte_ratio, 4))            # [5] low_one_byte_ratio
            current_result.append(round(average_encoding_time / 1e-6, 2))  # [6] average_encoding_time
            current_result.append(round(average_decoding_time / 1e-6, 2))  # [7] average_decoding_time
            current_result.append(round(error_ratio, 4))                   # [8] error_ratio
            current_result.append(memory_footprint)                        # [9] memory_footprint
            performance_results.append(current_result)

    
    save_res_flag = False
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
        print(f"{mpn:>{str_size[i]}}")
        #output_head += f"{mpn:<{str_size[i]}}"
    print(output_head)

    if save_res_flag:
        res_file.writelines(output_head + "\n")

    print(performance_results)
    for model_result in performance_results:
        cc_str = ""
        for i, v in enumerate(model_result):
            cc_str += f"{str(v):<{str_size[i]}}"
        print(cc_str)
        if save_res_flag:
            res_file.writelines(cc_str + "\n")

    if save_res_flag:
        res_file.close()
    '''

    os.system('sudo ifconfig can0 down')


def main():
    can0_send()


if __name__ == '__main__':
    main()