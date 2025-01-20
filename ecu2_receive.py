import os
import time
import can
from common import *
from my_model import *
import numpy as np


def can0_receive():

    os.system('sudo ip link set can0 type can bitrate 500000')
    os.system('sudo ifconfig can0 up')
    can0 = can.interface.Bus(channel='can0', bustype='socketcan')# socketcan_native

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
            step4_times_list = []
            step5_times_list = []
            updating_times_list = []
            latency_list = []
            latency2_list = []

            # 统计消息编解码错误的次数
            error_counts = 0

            # 统计读取文件的次数
            read_count = 0

            while True:

                msg = can0.recv(10.0)
                #print(msg)
                if msg is None:
                    print('Timeout occurred, no message.')
                else:

                    # read ground truth for error rate calculation
                    line_value = lines[read_count].split(',')
                    timestamp = int(line_value[0])
                    can_id = int(line_value[1], 16)
                    payload_str = line_value[2]
                    byte_data_str = payload_str.strip().split(' ')
                    byte_data = [x for x in byte_data_str]

                    #print(type(msg))
                    recv_can_id = msg.arbitration_id
                    recv_payload = list(msg.data)

                    message_str = ''.join([bin(value)[2:].zfill(8) for value in recv_payload])
                    #print(f"read_count: {read_count}, recv_can_id: {recv_can_id}, recv_payload: {recv_payload}")

                    # decoding the message
                    t4 = time.time()
                    complete_frame_flag, part_mark_flag, m_payload, mac_str = \
                        model.message_decoding_online(recv_can_id, message_str, minimum_mac_len)
                    step4_times_list.append(time.time() - t4)

                    # tag generation and verification
                    t5 = time.time()
                    content = ''.join(m_payload)
                    new_tag = hmac_can_message(secret_key, bytes.fromhex(content))
                    mac_length = len(mac_str)
                    new_mac_str = bin(int(new_tag, 16))[2:2 + mac_length]
                    #print(f"content: {content}")
                    step5_times_list.append(time.time() - t5)

                    verified_flag = True
                    if new_mac_str == mac_str:
                        if complete_frame_flag:
                            verified_flag = True
                            print(f"message correct, recv_tag:{mac_str}, new_tag: {new_mac_str}")
                        else:
                            # print(f"message part one correct, recv_tag:{mac_str}, new_tag: {new_mac_str}")

                            # recv for the partition two
                            msg2 = can0.recv(10.0)
                            if msg2 is None:
                                print('Timeout occurred, no message msg_part2.')
                            else:
                                recv_can_id2 = msg2.arbitration_id
                                recv_payload2 = list(msg2.data)

                                message_str2 = ''.join([bin(value)[2:].zfill(8) for value in recv_payload2])

                                # decoding the message
                                t4_2 = time.time()
                                complete_frame_flag2, part_mark_flag2, m_payload2, mac_str2 = \
                                    model.message_decoding_online(recv_can_id2, message_str2, minimum_mac_len)
                                step4_times_list.append(time.time() - t4_2)

                                t5_2 = time.time()
                                content2 = ''.join(m_payload2)
                                new_tag2 = hmac_can_message(secret_key, bytes.fromhex(content2))
                                mac_length2 = len(mac_str2)
                                new_mac_str2 = bin(int(new_tag2, 16))[2:2 + mac_length2]
                                step5_times_list.append(time.time() - t5_2)

                                if new_mac_str2 == mac_str2:
                                    print(f"message part two correct, recv_tag:{mac_str2}, new_tag: {new_mac_str2}")
                                    m_payload.extend(m_payload2)
                                else:
                                    verified_flag = False
                                    print(f"message error! recv_tag:{mac_str2}, new_tag: {new_mac_str2}")
                    else:
                        verified_flag = False
                        print(f"message error! recv_tag:{mac_str}, new_tag: {new_mac_str}")

                    #print(f"can_id: {can_id}, original_message: {byte_data}, decoding_payload: {m_payload}")

                    # update the dictionary
                    t6 = time.time()
                    if verified_flag:
                        model.dictionary_update(can_id, m_payload)
                    updating_time = time.time() - t6
                    updating_times_list.append(updating_time)

                    latency = time.time() - t4
                    if complete_frame_flag:
                        latency_list.append(latency)
                    else:
                        latency2_list.append(latency)

                    print(f"read_count: {read_count}, recv_can_id: {recv_can_id}, decoded_payload: {m_payload}")

                    # 判断是否编解码是否错误 for debug
                    for i in range(8):
                        if int(m_payload[i], 16) != int(byte_data[i], 16):
                            print("frame transmission error!!!")
                            print(f"decoded payload: {m_payload}")
                            print(f"actual payload: {byte_data}")
                            error_counts += 1
                            break

                    # for debug
                    read_count += 1
                    #if read_count > 50:
                    if read_count > 5000:
                    #if not complete_frame_flag:
                        break

            # calculate error ratio
            error_ratio = round(error_counts / read_count * 100, 2)
            print(f"read_count: {read_count}, error_counts: {error_counts}, error_ratio: {error_ratio}%")

            mean_step4_time = round(np.mean(step4_times_list) / 1e-6, 2)
            max_step4_time = round(np.max(step4_times_list) / 1e-6, 2)
            min_step4_time = round(np.min(step4_times_list) / 1e-6, 2)

            mean_step5_time = round(np.mean(step5_times_list) / 1e-6, 2)
            max_step5_time = round(np.max(step5_times_list) / 1e-6, 2)
            min_step5_time = round(np.min(step5_times_list) / 1e-6, 2)

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
            print(f"step4_time(us), mean: {mean_step4_time}, max: {max_step4_time}, min: {min_step4_time}")
            print(f"step5_time(us), mean: {mean_step5_time}, max: {max_step5_time}, min: {min_step5_time}")
            print(f"updating_time(us), mean: {mean_updating_time}, max: {max_updating_time}, min: {min_updating_time}")
            print(f"latency(us), mean: {mean_latency}, max: {max_latency}, min: {min_latency}")
            print(f"latency2(us), mean: {mean_latency2}, max: {max_latency2}, min: {min_latency2}")

    os.system('sudo ifconfig can0 down')


def main():
    can0_receive()


if __name__ == '__main__':
    main()