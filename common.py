import hmac
import hashlib
import sys
import time

secret_key = b'0123456789abcdef'
minimum_mac_len = 20

# 全局变量，文件路径
file_paths = [
    "dataset/can_intrusion/attack_free.csv",
    "dataset/car_hacking/attack_free.csv",
    #"dataset/survival/Sonata/attack_free.csv",    # vehicle repeat with car_hacking
    #"dataset/survival/Soul/attack_free.csv",      # vehicle repeat with can_intrusion
    "dataset/survival/Spark/attack_free.csv",
    "dataset/VTC2019Fall/clean/attack_free.csv",
    "dataset/can_intrusion_v2/data/OpelAstra/attack_free.csv",
    "dataset/can_intrusion_v2/data/RenaultClio/attack_free.csv",
    "dataset/can_signal/attack_free.csv",
    "dataset/ReCAN/C-1-AlfaRomeo-Giulia/Exp-1/attack_free.csv",
    "dataset/ReCAN/C-2-Opel-Corsa/Exp-1/attack_free.csv",
    "dataset/ReCAN/T-1-Mitsubishi-FusoCanter/Exp-1/attack_free.csv",
    "dataset/ReCAN/T-2-Isuzu-M55/Exp-1/attack_free.csv",
    "dataset/ReCAN/T-3-Piaggio-PorterMaxi/Exp-1/attack_free.csv",
    "dataset/car_hacking_challenge/0_Preliminary/0_Training/attack_free.csv",
    "dataset/heavy_duty_truck/part_1/attack_free.csv",
    "dataset/self_collection/001.csv",
    "dataset/road/ambient/attack_free.csv",
    "dataset/can_train_and_test/2011-chevrolet-impala/attack_free.csv",
    "dataset/can_train_and_test/2011-chevrolet-traverse/attack_free.csv",
    "dataset/can_train_and_test/2016-chevrolet-silverado/attack_free.csv",
    "dataset/can_train_and_test/2017-subaru-forester/attack_free.csv"
]

# 全局变量，汽车名称
vehicle_names = [
    "Kia Soul",
    "Hyundai YF Sonata",
    #"Hyundai YF Sonata",     # vehicle repeat with car_hacking
    #"Kia Soul",              # vehicle repeat with can_intrusion
    "Chevrolet Spark",
    "Volvo V40",
    "Opel Astra",
    "Renault Clio",
    "Unknown 1",
    "Alfa Romeo Giulia Veloce",
    "Opel Corsa",
    "Mitsubishi Fuso Canter",
    "Isuzu M55",
    "Piaggio Porter Maxi",
    "Hyundai Avante CN7",
    "Renault T520 6X2",
    "Yutong Xiaoyu",
    "Unknown 2",
    "Chevrolet Impala",
    "Chevrolet Traverse",
    "Chevrolet Silverado",
    "Subaru Forester",
]

# 全局变量，统计属性 mpn: models_performance_names
performance_names = [
    "Vehicle",
    "Methods",
    "min",
    "mean",
    "max",
    "Pcll",             # proportion of compression lower limit
    "Enc(us)",          # encoding time
    "Dec(us)",          # decoding time
    "error(%)",         # error_ratio
    "Mem(KB)"           # memory
]


# 获取变量的内存大小
def get_total_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])
    return size


def hmac_can_message(key, can_message):
    """
    使用 HMAC 算法计算 CAN 报文的认证标签

    参数:
    key (bytes) - 用于计算 HMAC 的密钥
    can_message (bytes) - CAN 报文的 8 字节数据

    返回:
    HMAC 认证标签 (bytes)
    """
    # 计算 HMAC 值
    hmac_value = hmac.new(key, can_message, hashlib.sha256).digest()

    # 转换为十六进制字符串
    auth_tag = hmac_value.hex()

    return auth_tag


def _double(block):
    """
    对 AES 块进行左移一位操作
    """
    result = b''.join(
        bytes([(byte << 1) & 0xFF, ((byte & 0x80) >> 7) * 0x87 & 0xFF])
        for byte in block
    )
    return result

def _xor(a, b):
    """
    对两个字节串进行 XOR 操作
    """
    return bytes(x ^ y for x, y in zip(a, b))


def calculate_hamming_distance(row1, row2):
    """计算两个报文之间的汉明距离"""
    data1 = row1['Payload']
    data2 = row2['Payload']
    #print("data1: ", data1)
    #print("data2: ", data2)

    data1 = data1.replace(' ', '')
    data2 = data2.replace(' ', '')
    #print("data1: ", data1)
    #print("data2: ", data2)

    #print("bin_data1: ", bin(int(data1, 16)))
    #print("bin_data2: ", bin(int(data2, 16)))

    distance = 0
    for i in range(len(data1)):
#        print("bin_data1: ", bin(int(data1[i], 16)))
#        print("bin_data2: ", bin(int(data2[i], 16)))
        if data1[i] != data2[i]:
            distance += bin(int(data1[i], 16) ^ int(data2[i], 16)).count('1')
#            print("distance: ", distance)
    return distance


# 计算字节距离,8个字节，字节相同即加1
def calculate_byte_distance(row1, row2):
    """计算两个报文之间的汉明距离"""
    # 首先,我们使用 strip() 方法去掉字符串两端的空白字符(包括前面的空格)。然后再使用 split() 方法按照空格将字符串拆分成列表。
    data1 = row1['Payload'].strip().split()
    data2 = row2['Payload'].strip().split()
    #print("data1: ", data1)
    #print("data2: ", data2)

    #data1 = data1.replace(' ', '')
    #data2 = data2.replace(' ', '')
    #print(f"data1_len: {len(data1)}, data1: {data1}")
    #print(f"data2_len: {len(data2)}, data1: {data2}")

    distance = 0
    for i in range(len(data1)):
#        print("bin_data1: ", bin(int(data1[i], 16)))
#        print("bin_data2: ", bin(int(data2[i], 16)))
        if data1[i] != data2[i]:
            distance += 1
#            print("distance: ", distance)
    return distance


def similarity_marker(data1, data2):

    similarity_str = ""
    for i in range(len(data1)):
#        print("bin_data1: ", bin(int(data1[i], 16)))
#        print("bin_data2: ", bin(int(data2[i], 16)))
        if data1[i] == data2[i]:
            similarity_str += "1"
        else:
            similarity_str += "0"
    return similarity_str

