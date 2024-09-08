import math
from collections import defaultdict


# table based data reduction
class TBDR:
    def __init__(self, block_size=8, read_ratio=0.1):
        self.dictionary = None                            # 存储映射字典
        self.block_size = block_size                      # 64个bit位切分的块大小
        self.read_ratio = read_ratio                      # 训练字典的数据比例,取值[0, 1]

    # 根据数据获取静态字典
    def dictionary_building(self, data_str):
        """
        :param data_str:      字符串数据
        :return:
        """

        read_ratio = self.read_ratio
        bit_length = self.block_size

        lines = data_str.strip().split("\n")
        total_number = len(lines)
        end_pos = int(total_number * read_ratio)
        lines = lines[:end_pos]

        # 初始化存储变量
        bits_static_dictionary = defaultdict(list)  # 存储字节的取值  "ID=0x6aa,byte=1" = [0,56]


        # 第一次处理数据，获取静态字典
        reports = []
        for line in lines:
            timestamp, can_id, payload = line.split(",")
            can_id = int(can_id, 16)

            payload_bytes = [x for x in payload.strip().split()]
            # 将负载值转化成64位的数值
            payload_bin_str = ""
            for x in payload_bytes:
                # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                payload_bin_str += bin(int(x, 16))[2:].zfill(8)
                # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

            # 根据划分的长度拆分负载
            bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

            # print(f"payload: {payload}")
            # print(f"payload_bin_str: {payload_bin_str}, len: {len(payload_bin_str)} bits")
            # print(f"bits_list: {bits_list}, len: {len(bits_list)} blocks")

            reports.append((can_id, bits_list))

        # 统计每个字节可能出现的值个数, defaultdict默认字典，不会引发错误值
        result = defaultdict(lambda: [0] * pow(2, bit_length))
        for can_id, bits_list in reports:
            for block_pos, bits_value in enumerate(bits_list):
                result[(can_id, block_pos)][bits_value] += 1

        # 输出结果
        for (can_id, block_pos), counts in result.items():
            # print(f"ID: 0x{can_id:X}, block_pos: {block_pos}")
            total_count = len([count for count in counts if count > 0])
            # print(f"  total number: {total_count}")
            bits_values = []
            for j, count in enumerate(counts):
                if count > 0:
                    bits_values.append((j, count))
            bits_values.sort(key=lambda x: x[1], reverse=True)

            bits_static_dictionary[(can_id, block_pos)] = [x[0] for x in bits_values]

            # for bits_value, count in bits_values:
            #    print(f"  bits value {bits_value:0X}: {count}")
            # print()

        self.dictionary = bits_static_dictionary

    def message_encoding(self, m_id, m_payload):
        """
        根据字典对消息进行压缩
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        # encoding the message
        encoding_message_str = ""
        list_bit_len = []
        compression_marker_str = ""  # 是否压缩的标记位

        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in m_payload:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 根据划分的长度拆分负载
        bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

        for i, bits_value in enumerate(bits_list):

            # 1.读取相应的字典映射码
            bits_value_dictionary = dictionary[(m_id, i)]

            find_index = -1  # 字节值在字典映射中的索引
            bit_len = -1  # 当前字节采用压缩模式占用的比特位数
            binary_index = -1  # 字节值在字典映射中的索引-压缩后的形式

            #print(
            #    f"ID: {m_id:x}, pos: {i}, bits_value: {bits_value}, byte_value_dictionary: {bits_value_dictionary}")

            if bits_value not in bits_value_dictionary:
                find_index = -1
                bit_len = bit_length
                binary_index = -1
                compression_marker_str += "0"  # 无法压缩
                encoding_message_str += bin(bits_value)[2:].zfill(bit_len)

            else:
                find_index = bits_value_dictionary.index(bits_value)  # 字节值在字典映射中的索引
                bit_len = math.ceil(math.log2(len(bits_value_dictionary)))
                compression_marker_str += "1"  # 可以压缩

                if len(bits_value_dictionary) == 1:
                    # 如果字节值取值可能只有1种，不需要占用额外空间
                    encoding_message_str += ""
                else:
                    # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                    # # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                    binary_index = bin(find_index)[2:].zfill(bit_len)
                    encoding_message_str += binary_index

            #print(f"find_index: {find_index}, binary_index: {binary_index}, bit_len: {bit_len}")
            list_bit_len.append(bit_len)

        message_str = compression_marker_str + encoding_message_str

        #print(f"compression_marker_str: {compression_marker_str}")
        #print(f"list_bit_len: {list_bit_len}")
        #print(f"encoding_message_str: {encoding_message_str}")
        #print(f"message_str: {message_str}")

        return message_str

    def message_decoding(self, m_id, message_str):
        """
        根据字典对接收的消息进行解压缩
        :param m_id:        消息ID
        :param message_str: 编码形式的二进制序列
        :return: 原始消息, MAC
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        # 根据划分的长度计算位图长度(bits)
        bitmap_length = int(64 / bit_length)

        bitmap_header = message_str[0:bitmap_length]
        cc_message_str = message_str[bitmap_length:]
        bit_len = []
        m_payload_bin_str = ""

        #print(f"bitmap_header: {bitmap_header}, cc_message_str: {cc_message_str}")

        for i, bit_value in enumerate(bitmap_header):
            if bit_value == "0":
                # 不可压缩
                cc_bit_len = bit_length
                bit_len.append(cc_bit_len)

                cc_bits_value = cc_message_str[0:cc_bit_len]
                m_payload_bin_str += cc_bits_value

                cc_message_str = cc_message_str[cc_bit_len:]
            else:
                # 可以压缩
                # 1.读取相应的字典映射码
                bits_value_dictionary = dictionary[(m_id, i)]
                # print(f"ID: {m_id:x}, pos: {i}, bits_value_dictionary: {bits_value_dictionary}")

                cc_bit_len = math.ceil(math.log2(len(bits_value_dictionary)))
                bit_len.append(cc_bit_len)

                if cc_bit_len == 0:
                    # 字典取值数目仅为1
                    cc_bits_value = bits_value_dictionary[0]
                    m_payload_bin_str += bin(cc_bits_value)[2:].zfill(bit_length)
                else:
                    cc_bits_value_index_str = cc_message_str[0:cc_bit_len]
                    # 字典值索引
                    cc_bits_value_index = int(cc_bits_value_index_str, 2)
                    # 字典值
                    cc_bits_value = bits_value_dictionary[cc_bits_value_index]
                    m_payload_bin_str += bin(cc_bits_value)[2:].zfill(bit_length)

                    cc_message_str = cc_message_str[cc_bit_len:]

        # 固定转化为8字节的形式，使用内置的 hex() 函数将 10 进制数转换为16进制字符串。这个函数会在结果前面加上 "0x" 前缀,所以我们使用 [2:] 来去除前缀
        m_payload = [hex(int(m_payload_bin_str[i:i + 8], 2))[2:].zfill(2) for i in range(0, len(m_payload_bin_str), 8)]

        return m_payload

    def dictionary_update(self, m_id, m_payload):
        """
        根据接收的消息对字典进行更新
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           无
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        payload_bytes = [x for x in m_payload]
        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in payload_bytes:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 根据划分的长度拆分负载
        bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

        for i, bits_value in enumerate(bits_list):
            # 读取相应的字典映射码
            bits_value_dictionary = dictionary[(m_id, i)]

            if bits_value not in bits_value_dictionary:
                # 静态字典动态更新
                dictionary[(m_id, i)].append(bits_value)

        self.dictionary = dictionary


# improved table based data reduction
class ITBDR:
    def __init__(self, block_size=8, read_ratio=0.1, silence_threshold=6):
        self.dictionary = None                            # 存储映射字典
        self.block_size = block_size                      # 64个bit位切分的块大小
        self.read_ratio = read_ratio                      # 训练字典的数据比例,取值[0, 1]
        self.silence_threshold = silence_threshold        # 优化时活跃值数目所占位数的阈值

    # 根据数据获取静态字典
    def dictionary_building(self, data_str):
        """
        :param data_str:      字符串数据
        :return:
        """

        read_ratio = self.read_ratio
        bit_length = self.block_size
        silence_threshold = self.silence_threshold

        lines = data_str.strip().split("\n")
        total_number = len(lines)
        end_pos = int(total_number * read_ratio)
        lines = lines[:end_pos]

        # 初始化存储变量
        bits_static_dictionary = defaultdict(list)  # 存储字节的取值  "ID=0x6aa,byte=1" = [0,56]

        # 第一次处理数据，获取静态字典
        reports = []
        for line in lines:
            timestamp, can_id, payload = line.split(",")
            can_id = int(can_id, 16)

            payload_bytes = [x for x in payload.strip().split()]
            # 将负载值转化成64位的数值
            payload_bin_str = ""
            for x in payload_bytes:
                # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                payload_bin_str += bin(int(x, 16))[2:].zfill(8)
                # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

            # 根据划分的长度拆分负载
            bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

            # print(f"payload: {payload}")
            # print(f"payload_bin_str: {payload_bin_str}, len: {len(payload_bin_str)} bits")
            # print(f"bits_list: {bits_list}, len: {len(bits_list)} blocks")

            reports.append((can_id, bits_list))

        # 统计每个字节可能出现的值个数, defaultdict默认字典，不会引发错误值
        result = defaultdict(lambda: [0] * pow(2, bit_length))
        for can_id, bits_list in reports:
            for block_pos, bits_value in enumerate(bits_list):
                result[(can_id, block_pos)][bits_value] += 1

        # 输出结果
        for (can_id, block_pos), counts in result.items():
            # print(f"ID: 0x{can_id:X}, block_pos: {block_pos}")
            total_count = len([count for count in counts if count > 0])
            # print(f"  total number: {total_count}")
            bits_values = []
            for j, count in enumerate(counts):
                if count > 0:
                    bits_values.append((j, count))
            bits_values.sort(key=lambda x: x[1], reverse=True)

            bits_static_dictionary[(can_id, block_pos)] = [x[0] for x in bits_values]

            # 如果取值数目大于(固定值)上限的一半，则没有必要存储了，并用-1标记
            if len(bits_static_dictionary[(can_id, block_pos)]) > pow(2, silence_threshold):
                bits_static_dictionary[(can_id, block_pos)] = []
                bits_static_dictionary[(can_id, block_pos)].append(-1)

            # for bits_value, count in bits_values:
            #    print(f"  bits value {bits_value:0X}: {count}")
            # print()

        self.dictionary = bits_static_dictionary

    def message_encoding(self, m_id, m_payload):
        """
        根据字典对消息进行压缩
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        # encoding the message
        encoding_message_str = ""
        list_bit_len = []
        compression_marker_str = ""  # 是否压缩的标记位

        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in m_payload:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 根据划分的长度拆分负载
        bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

        for i, bits_value in enumerate(bits_list):

            # 1.读取相应的字典映射码
            bits_value_dictionary = dictionary[(m_id, i)]

            find_index = -1  # 字节值在字典映射中的索引
            bit_len = -1  # 当前字节采用压缩模式占用的比特位数
            binary_index = -1  # 字节值在字典映射中的索引-压缩后的形式

            #print(
            #    f"ID: {m_id:x}, pos: {i}, bits_value: {bits_value}, byte_value_dictionary: {bits_value_dictionary}")

            if bits_value not in bits_value_dictionary:
                find_index = -1
                bit_len = bit_length
                binary_index = -1

                # 确定是否被忽略
                ignore_flag = False
                if len(bits_value_dictionary) == 1:
                    # -1表示该值已经被放弃
                    if bits_value_dictionary[0] == -1:
                        ignore_flag = True

                if ignore_flag:
                    compression_marker_str += ""  # 无法压缩,且压缩标记被忽略
                else:
                    compression_marker_str += "0"  # 无法压缩

                encoding_message_str += bin(bits_value)[2:].zfill(bit_len)

            else:
                find_index = bits_value_dictionary.index(bits_value)  # 字节值在字典映射中的索引
                bit_len = math.ceil(math.log2(len(bits_value_dictionary)))
                compression_marker_str += "1"  # 可以压缩

                if len(bits_value_dictionary) == 1:
                    # 如果字节值取值可能只有1种，不需要占用额外空间
                    encoding_message_str += ""
                else:
                    # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                    # # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                    binary_index = bin(find_index)[2:].zfill(bit_len)
                    encoding_message_str += binary_index

            #print(f"find_index: {find_index}, binary_index: {binary_index}, bit_len: {bit_len}")
            list_bit_len.append(bit_len)

        message_str = compression_marker_str + encoding_message_str

        #print(f"compression_marker_str: {compression_marker_str}")
        #print(f"list_bit_len: {list_bit_len}")
        #print(f"encoding_message_str: {encoding_message_str}")
        #print(f"message_str: {message_str}")

        return message_str

    def message_decoding(self, m_id, message_str):
        """
        根据字典对接收的消息进行解压缩
        :param m_id:        消息ID
        :param message_str: 编码形式的二进制序列
        :return: 原始消息, MAC
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        # 计算块数目
        block_quantity = int(64 / bit_length)

        # 查找被忽略的标记位
        ignore_pos = []
        for pos in range(block_quantity):
            bits_value_dictionary = dictionary[(m_id, pos)]
            # 确定是否被忽略
            if len(bits_value_dictionary) == 1:
                # -1表示该值已经被放弃
                if bits_value_dictionary[0] == -1:
                    ignore_pos.append(pos)

        # 计算余下的位图长度
        bitmap_length = block_quantity - len(ignore_pos)

        bitmap_header = message_str[0:bitmap_length]
        cc_message_str = message_str[bitmap_length:]
        bit_len = []
        m_payload_bin_str = ""

        #print(f"bitmap_header: {bitmap_header}, cc_message_str: {cc_message_str}")

        bitmap_header_pos = 0
        for pos in range(block_quantity):
            if pos in ignore_pos:
                # 不可压缩
                cc_bit_len = bit_length
                bit_len.append(cc_bit_len)

                cc_bits_value = cc_message_str[0:cc_bit_len]
                m_payload_bin_str += cc_bits_value

                cc_message_str = cc_message_str[cc_bit_len:]
            else:
                bit_value = bitmap_header[bitmap_header_pos]
                bitmap_header_pos = bitmap_header_pos + 1

                if bit_value == "0":
                    # 不可压缩
                    cc_bit_len = bit_length
                    bit_len.append(cc_bit_len)

                    cc_bits_value = cc_message_str[0:cc_bit_len]
                    m_payload_bin_str += cc_bits_value

                    cc_message_str = cc_message_str[cc_bit_len:]
                else:
                    # 可以压缩
                    # 1.读取相应的字典映射码
                    bits_value_dictionary = dictionary[(m_id, pos)]
                    # print(f"ID: {m_id:x}, pos: {i}, bits_value_dictionary: {bits_value_dictionary}")

                    cc_bit_len = math.ceil(math.log2(len(bits_value_dictionary)))
                    bit_len.append(cc_bit_len)

                    if cc_bit_len == 0:
                        # 字典取值数目仅为1
                        cc_bits_value = bits_value_dictionary[0]
                        m_payload_bin_str += bin(cc_bits_value)[2:].zfill(bit_length)
                    else:
                        cc_bits_value_index_str = cc_message_str[0:cc_bit_len]
                        # 字典值索引
                        cc_bits_value_index = int(cc_bits_value_index_str, 2)
                        # 字典值
                        cc_bits_value = bits_value_dictionary[cc_bits_value_index]
                        m_payload_bin_str += bin(cc_bits_value)[2:].zfill(bit_length)

                        cc_message_str = cc_message_str[cc_bit_len:]

        # 固定转化为8字节的形式，使用内置的 hex() 函数将 10 进制数转换为16进制字符串。这个函数会在结果前面加上 "0x" 前缀,所以我们使用 [2:] 来去除前缀
        m_payload = [hex(int(m_payload_bin_str[i:i + 8], 2))[2:].zfill(2) for i in range(0, len(m_payload_bin_str), 8)]

        return m_payload

    def dictionary_update(self, m_id, m_payload):
        """
        根据接收的消息对字典进行更新（使用模型更新，压缩更稳定）
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           无
        """

        dictionary = self.dictionary
        bit_length = self.block_size
        silence_threshold = self.silence_threshold

        payload_bytes = [x for x in m_payload]
        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in payload_bytes:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 根据划分的长度拆分负载
        bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

        for block_pos, bits_value in enumerate(bits_list):
            # 读取相应的字典映射码
            bits_value_dictionary = dictionary[(m_id, block_pos)]

            if bits_value not in bits_value_dictionary:

                # 判断该位置是否被忽略
                ignore_flag = False
                if len(bits_value_dictionary) == 1:
                    # -1表示该值已经被放弃
                    if bits_value_dictionary[0] == -1:
                        ignore_flag = True

                if ignore_flag is False:
                    # 字典动态更新
                    dictionary[(m_id, block_pos)].append(bits_value)

                    # 如果取值数目大于(固定值)上限的一半，则没有必要存储了，并用-1标记
                    if len(dictionary[(m_id, block_pos)]) > pow(2, silence_threshold):
                        dictionary[(m_id, block_pos)] = []
                        dictionary[(m_id, block_pos)].append(-1)

        self.dictionary = dictionary

    def message_encoding_with_detailed_output(self, m_id, m_payload):
        """
        根据字典对消息进行压缩，同时返回标记位大小、每个块压缩后的大小
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        dictionary = self.dictionary
        bit_length = self.block_size

        # encoding the message
        encoding_message_str = ""
        list_bit_len = []
        compression_marker_str = ""  # 是否压缩的标记位

        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in m_payload:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 根据划分的长度拆分负载
        bits_list = [int(payload_bin_str[i:i + bit_length], 2) for i in range(0, len(payload_bin_str), bit_length)]

        for i, bits_value in enumerate(bits_list):

            # 1.读取相应的字典映射码
            bits_value_dictionary = dictionary[(m_id, i)]

            find_index = -1  # 字节值在字典映射中的索引
            bit_len = -1  # 当前字节采用压缩模式占用的比特位数
            binary_index = -1  # 字节值在字典映射中的索引-压缩后的形式

            #print(
            #    f"ID: {m_id:x}, pos: {i}, bits_value: {bits_value}, byte_value_dictionary: {bits_value_dictionary}")

            if bits_value not in bits_value_dictionary:
                find_index = -1
                bit_len = bit_length
                binary_index = -1

                # 确定是否被忽略
                ignore_flag = False
                if len(bits_value_dictionary) == 1:
                    # -1表示该值已经被放弃
                    if bits_value_dictionary[0] == -1:
                        ignore_flag = True

                if ignore_flag:
                    compression_marker_str += ""  # 无法压缩,且压缩标记被忽略
                else:
                    compression_marker_str += "0"  # 无法压缩

                encoding_message_str += bin(bits_value)[2:].zfill(bit_len)

            else:
                find_index = bits_value_dictionary.index(bits_value)  # 字节值在字典映射中的索引
                bit_len = math.ceil(math.log2(len(bits_value_dictionary)))
                compression_marker_str += "1"  # 可以压缩

                if len(bits_value_dictionary) == 1:
                    # 如果字节值取值可能只有1种，不需要占用额外空间
                    encoding_message_str += ""
                else:
                    # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                    # # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                    binary_index = bin(find_index)[2:].zfill(bit_len)
                    encoding_message_str += binary_index

            #print(f"find_index: {find_index}, binary_index: {binary_index}, bit_len: {bit_len}")
            list_bit_len.append(bit_len)

        message_str = compression_marker_str + encoding_message_str

        #print(f"compression_marker_str: {compression_marker_str}")
        #print(f"list_bit_len: {list_bit_len}")
        #print(f"encoding_message_str: {encoding_message_str}")
        #print(f"message_str: {message_str}")

        return message_str, compression_marker_str, list_bit_len



# 时序冗余方法，第一字节标记位
class Baseline:
    def __init__(self):
        self.message_cache = {}                         # 存储上一次报文

    def message_encoding(self, m_id, m_payload):
        """
        根据字典对消息进行压缩
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        message_cache = self.message_cache

        # encoding the message
        encoding_message_str = ""
        compression_marker_str = ""  # 是否压缩的标记位

        if m_id not in message_cache:
            # first message
            for i in range(len(m_payload)):
                compression_marker_str += "0"
                encoding_message_str += bin(int(m_payload[i], 16))[2:].zfill(8)
        else:
            # not the first message
            payload1 = m_payload
            payload2 = message_cache[m_id]

            # for x in m_payload
            for i in range(len(payload1)):
                #        print("bin_data1: ", bin(int(data1[i], 16)))
                #        print("bin_data2: ", bin(int(data2[i], 16)))
                if payload1[i] == payload2[i]:
                    compression_marker_str += "1"
                    encoding_message_str += ""
                else:
                    compression_marker_str += "0"
                    encoding_message_str += bin(int(payload1[i], 16))[2:].zfill(8)

        message_str = compression_marker_str + encoding_message_str

        return message_str

    def message_decoding(self, m_id, message_str):
        '''
        对压缩的消息进行解码
        '''

        message_cache = self.message_cache

        bitmap_header = message_str[0:8]
        cc_message_str = message_str[8:]
        bit_len = []
        m_payload_bin_str = ""

        for i, bit_value in enumerate(bitmap_header):
            if bit_value == "0":
                # 不可压缩
                cc_bits_value = cc_message_str[0:8]
                m_payload_bin_str += cc_bits_value

                cc_message_str = cc_message_str[8:]
            else:
                # 可以压缩
                # 读取相应的缓存
                previous_payload = message_cache[m_id]

                m_payload_bin_str += bin(int(previous_payload[i], 16))[2:].zfill(8)

        # 固定转化为8字节的形式，使用内置的 hex() 函数将 10 进制数转换为16进制字符串。这个函数会在结果前面加上 "0x" 前缀,所以我们使用 [2:] 来去除前缀
        m_payload = [hex(int(m_payload_bin_str[i:i + 8], 2))[2:].zfill(2) for i in range(0, len(m_payload_bin_str), 8)]

        return m_payload

    def message_cache_update(self, m_id, m_payload):
        """
        根据接收的消息对缓存进行更新
        :param m_id:       消息ID
        :param m_payload:  消息负载
        """
        message_cache = self.message_cache

        # update the message_cache
        message_cache[m_id] = m_payload

        self.message_cache = message_cache


# 时序冗余压缩，计算前后报文的delta差值
class CASmap:
    def __init__(self):
        self.message_cache = {}                         # 存储上一次报文

    @staticmethod
    def convert_into_three_signal(m_payload):
        # 将8字节负载转换成24，24，16比特位的3个信号
        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in m_payload:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        # 将2进制负载重新管理成24,24,16比特位的3个信号
        signals_bin_str = [payload_bin_str[:24], payload_bin_str[24:48], payload_bin_str[48:]]
        return signals_bin_str

    def message_encoding(self, m_id, m_payload):
        """
        根据字典对消息进行压缩
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        message_cache = self.message_cache

        # encoding the message
        encoding_message_str = ""
        header_map_str = ""  # 是否压缩的标记位

        # 将2进制负载重新管理成24,24,16比特位的3个信号
        signals_bin_str = self.convert_into_three_signal(m_payload)

        if m_id not in message_cache:
            # first message
            for i in range(len(signals_bin_str)):
                header_map_str += "0"
                encoding_message_str += signals_bin_str[i]
        else:
            # Not the first message
            current_payload = m_payload
            previous_payload = message_cache[m_id]

            # 24,24,16格式的3个信号
            previous_signals_bin_str = self.convert_into_three_signal(previous_payload)
            current_signals_bin_str = self.convert_into_three_signal(current_payload)

            #计算前后两个信号之间的异或值
            signals_xor_result = []
            for i in range(len(current_signals_bin_str)):
                a_bin_str = current_signals_bin_str[i]
                b_bin_str = previous_signals_bin_str[i]

                a_bin_list = list(a_bin_str)
                b_bin_list = list(b_bin_str)
                xor_bin_list = []

                for pos in range(len(a_bin_list)):
                    bit_xor = int(a_bin_list[pos], 2) ^ int(b_bin_list[pos], 2)
                    xor_bin_list.append(str(bit_xor))

                xor_bin_str = "".join(xor_bin_list)
                signals_xor_result.append(xor_bin_str)

            # debug: 打印2进制信号以及异或结果
            #print(f"message encoding:")
            #for i in range(len(current_signals_bin_str)):
                #print(f"current_signals_bin_str[{i}]: {current_signals_bin_str[i]}")
                #print(f"previous_signals_bin_str[{i}]: {previous_signals_bin_str[i]}")
                #print(f"signals_xor_result[{i}]: {list(signals_xor_result[i])}")

            # 根据异或结果确定header的值
            for i in range(len(signals_xor_result)):
                xor_bin_str = signals_xor_result[i]
                xor_decimal = int(xor_bin_str, 2)
                if xor_decimal == 0:
                    header_map_str += "0"
                else:
                    header_map_str += "1"

            # 寻找3个异或字符串的最大非0索引
            max_last_pos_index = -1
            for i in range(len(signals_xor_result)):
                if header_map_str[i] == "0":
                    continue
                else:
                    xor_bin_str = signals_xor_result[i]
                    last_pos_index = -1
                    # 从后向前遍历字符串,找到第一个非零字符的位置
                    for j in range(len(xor_bin_str) - 1, -1, -1):
                        if xor_bin_str[j] != "0":
                            last_pos_index = j
                            break

                    max_last_pos_index = max(max_last_pos_index, last_pos_index)
            #print(f"max_last_pos_index: {max_last_pos_index}")

            for i in range(max_last_pos_index+1):
                for signal_num in range(len(header_map_str)):
                    if header_map_str[signal_num] == "0":
                        continue
                    else:
                        # signal A，B 有3个字节，signal C 只有2个字节，这里防止超出signal C的边界
                        if i >= len(signals_bin_str[signal_num]):
                            continue
                        else:
                            encoding_message_str += signals_xor_result[signal_num][i]

        message_str = header_map_str + encoding_message_str

        return message_str

    def message_decoding(self, m_id, message_str):
        '''
        对压缩的消息进行解码
        '''

        message_cache = self.message_cache
        m_payload_bin_str = ""

        header_map_str = message_str[0:3]
        encoding_message_str = message_str[3:]

        #print(f"message decoding:")
        #print(f"header_map_str: {header_map_str}, encoding_message_str: {encoding_message_str}")

        if m_id not in message_cache:
            # first message
            m_payload_bin_str = encoding_message_str
        else:
            # Not the first message
            signal_len = [24, 24, 16]
            signals_xor_result = [[], [], []]
            value_pos = 0
            # 根据CAS map恢复3个信号的异或值
            while value_pos < len(encoding_message_str):
                for signal_num in range(len(header_map_str)):
                    if header_map_str[signal_num] == "0":
                        continue
                    else:
                        if len(signals_xor_result[signal_num]) < signal_len[signal_num]:
                            signals_xor_result[signal_num].append(encoding_message_str[value_pos])
                            value_pos += 1

            # 对信号的异或值进行填充，恢复为24，24，16
            for i in range(len(signal_len)):
                # 补齐负载省略的0
                #print(f"signals_xor_list len[{i}]: {len(signals_xor_list)}")
                # 如果长度超过信号本身长度，则不需要填充
                if len(signals_xor_result[i]) < signal_len[i]:
                    for j in range(signal_len[i] - len(signals_xor_result[i])):
                        signals_xor_result[i].append("0")
                #print(f"signals_xor_list[{i}]: {signals_xor_result[i]}")

            # 负载解码 original_value = cache_value ^ xor_result
            previous_payload = message_cache[m_id]
            previous_signals_bin_str = self.convert_into_three_signal(previous_payload)

            # 计算前后两个信号之间的异或值
            current_payload = []
            for i in range(len(signal_len)):
                a_bin_list = signals_xor_result[i]
                b_bin_list = list(previous_signals_bin_str[i])
                signal_bin_list = []

                #print(f"a_bin_list: {a_bin_list}, \nb_bin_list: {b_bin_list}")

                for pos in range(len(a_bin_list)):
                    bit_xor = int(a_bin_list[pos], 2) ^ int(b_bin_list[pos], 2)
                    signal_bin_list.append(str(bit_xor))
                #print(f"signal_bin_list: {signal_bin_list}")

                signal_bin_str = "".join(signal_bin_list)
                current_payload.append(signal_bin_str)

            m_payload_bin_str = "".join(current_payload)

        # 固定转化为8字节的形式，使用内置的 hex() 函数将 10 进制数转换为16进制字符串。这个函数会在结果前面加上 "0x" 前缀,所以我们使用 [2:] 来去除前缀
        m_payload = [hex(int(m_payload_bin_str[i:i + 8], 2))[2:].zfill(2) for i in range(0, len(m_payload_bin_str), 8)]

        return m_payload

    def message_cache_update(self, m_id, m_payload):
        """
        根据接收的消息对缓存进行更新
        :param m_id:       消息ID
        :param m_payload:  消息负载（8字节形式）
        """
        message_cache = self.message_cache

        # update the message_cache
        message_cache[m_id] = m_payload

        self.message_cache = message_cache


# Single-Frame-Based Data Compression
class SFDC:
    def __init__(self, read_ratio=0.1):
        self.mapping_matrix = None                        # 存储映射转换矩阵
        self.inversion_pos = None                         # 存储需要反转的位置
        self.read_ratio = read_ratio                      # 训练字典的数据比例

    # 根据数据建立映射矩阵
    def mapping_matrix_building(self, data_str):
        """
        :param data_str:      字符串数据
        :return:
        """

        read_ratio = self.read_ratio

        lines = data_str.strip().split("\n")
        total_number = len(lines)
        end_pos = int(total_number * read_ratio)
        lines = lines[:end_pos]

        # 第一次处理数据，获取静态字典
        reports = []
        message_count = {}   # 统计每个ID的报文数目
        id_list = []
        for line in lines:
            timestamp, can_id, payload = line.split(",")
            can_id = int(can_id, 16)

            if can_id not in id_list:
                id_list.append(can_id)

            payload_bytes = [x for x in payload.strip().split()]
            # 将负载值转化成64位的数值
            payload_bin_str = ""
            for x in payload_bytes:
                # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
                # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
                payload_bin_str += bin(int(x, 16))[2:].zfill(8)
                # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")
            reports.append((can_id, payload_bin_str))

            if can_id not in message_count:
                message_count[can_id] = 1
            else:
                message_count[can_id] += 1

        # Debug
        #print("message_count: ", message_count)

        # 统计每个比特位出现1的次数, defaultdict默认字典，不会引发错误值
        result = defaultdict(lambda: [0] * 64)
        for can_id, payload_bin_str in reports:
            for bit_pos in range(len(payload_bin_str)):
                if payload_bin_str[bit_pos] == "1":
                    result[can_id][bit_pos] += 1

        # 输出结果
        #print("result: ", result)
        #for can_id, counts in result.items():
        #    print(f"ID: 0x{can_id:X}, counts: {counts}")

        # 统计每个比特位出现1的概率, defaultdict默认字典，不会引发错误值
        for can_id in id_list:
            for bit_pos in range(64):
                result[can_id][bit_pos] = result[can_id][bit_pos]/message_count[can_id]

        # 输出结果
        #for can_id, probability in result.items():
        #    print(f"ID: 0x{can_id:X}, probability: {probability}")

        # 对概率大于50%的位置进行反转
        inversion_pos = defaultdict(list)                # 存储需要反转的位置
        for can_id in id_list:
            for bit_pos in range(64):
                if result[can_id][bit_pos] > 0.5:
                    result[can_id][bit_pos] = 1-result[can_id][bit_pos]
                    inversion_pos[can_id].append(bit_pos)

        # 输出结果
        #for can_id, probability in result.items():
        #    print(f"ID: 0x{can_id:X}, inversion_pos: {inversion_pos[can_id]}")
        #    print(f"ID: 0x{can_id:X}, probability: {probability}")

        # 对概率矩阵从大到小降序排列，并记录映射位置
        mapping_matrix = defaultdict(list)               # 存储排序映射
        for can_id in id_list:
            original_list = result[can_id].copy()

            #for i in range(64):
            #    max_value, max_index = max((v, i) for i, v in enumerate(original_list))
            #    mapping_matrix[can_id].append(max_index)
            #    original_list[max_index] = -1        # 将已使用过的值设为 -1 (消除列表中存在重复值的影响)

            # 对列表进行降序排列
            sorted_list = sorted(original_list, reverse=True)

            # 记录排序后的数据对应原始数据的位置
            for value in sorted_list:
                index = next((i for i, v in enumerate(original_list) if v == value), None)
                if index is not None:
                    mapping_matrix[can_id].append(index)
                    original_list[index] = None  # 将已使用过的值设为 None (消除列表中存在重复值的影响)
            # 输出结果
            #print(f"ID: 0x{can_id:X}, index_mapping: {mapping_matrix[can_id]}")
            #print(f"ID: 0x{can_id:X}, probability: {sorted_list}")

        self.mapping_matrix = mapping_matrix
        self.inversion_pos = inversion_pos

    def message_encoding(self, m_id, m_payload):
        """
        根据字典对消息进行压缩
        :param dictionary: 字节取值字典
        :param m_id:       消息ID
        :param m_payload:  消息负载
        :return:           压缩后的消息(64bits)
        """

        mapping_matrix = self.mapping_matrix
        inversion_pos = self.inversion_pos

        # 将负载值转化成64位的数值
        payload_bin_str = ""
        for x in m_payload:
            # 使用内置的 bin() 函数将 10 进制数转换为二进制字符串。这个函数会在结果前面加上 "0b" 前缀,所以我们使用 [2:] 来去除前缀
            # 使用 zfill(8) 方法将二进制字符串填充到 8 位长度。如果原始二进制数小于 8 位,会在左侧补 0
            payload_bin_str += bin(int(x, 16))[2:].zfill(8)
            # print(f"x: {x}, bin_x: {bin(int(x, 16))[2:]}")

        current_inversion_pos = inversion_pos[m_id]
        current_mapping_index = mapping_matrix[m_id]

        # 二进制序列对应比特位反转
        # 将字符串转换为列表(字符串不支持直接修改，因此先转换成列表，修改后，再转回字符串)
        inverse_payload_bin_list = list(payload_bin_str)
        # 修改指定位置的值
        for bit_pos in current_inversion_pos:
            if inverse_payload_bin_list[bit_pos] == "1":
                inverse_payload_bin_list[bit_pos] = "0"
            else:
                inverse_payload_bin_list[bit_pos] = "1"

        # 二进制序列映射
        mapping_payload_bin_list = []
        for mapping_index in current_mapping_index:
            mapping_payload_bin_list.append(inverse_payload_bin_list[mapping_index])

        # 将列表转回字符串
        mapping_payload_bin_list = ''.join(mapping_payload_bin_list)

        # 从后向前遍历字符串,找到第一个非零字符的位置
        last_pos_index = -1
        for i in range(len(mapping_payload_bin_list) - 1, -1, -1):
            if mapping_payload_bin_list[i] != "0":
                last_pos_index = i
                break

        message_str = ""
        if last_pos_index >= 0:
            message_str = "" + mapping_payload_bin_list[:last_pos_index+1]

        return message_str

    def message_decoding(self, m_id, message_str):
        """
        根据字典对接收的消息进行解压缩
        :param m_id:        消息ID
        :param message_str: 编码形式的二进制序列
        :return: 原始消息, MAC
        """

        mapping_matrix = self.mapping_matrix
        inversion_pos = self.inversion_pos

        current_inversion_pos = inversion_pos[m_id]
        current_mapping_index = mapping_matrix[m_id]

        # 补齐负载省略的0
        payload_bin_list = list(message_str)
        for i in range(64-len(payload_bin_list)):
            payload_bin_list.append("0")

        # 反向映射
        re_mapping_payload_bin_list = []
        for bit_pos in range(64):
            recover_index = current_mapping_index.index(bit_pos)
            re_mapping_payload_bin_list.append(payload_bin_list[recover_index])

        # 特定比特位反转，修改指定位置的值
        inverse_payload_bin_list = re_mapping_payload_bin_list
        for bit_pos in current_inversion_pos:
            if inverse_payload_bin_list[bit_pos] == "1":
                inverse_payload_bin_list[bit_pos] = "0"
            else:
                inverse_payload_bin_list[bit_pos] = "1"

        # 将列表转回字符串
        inverse_payload_bin_str = ''.join(inverse_payload_bin_list)

        m_payload_bin_str = inverse_payload_bin_str

        # 固定转化为8字节的形式，使用内置的 hex() 函数将 10 进制数转换为16进制字符串。这个函数会在结果前面加上 "0x" 前缀,所以我们使用 [2:] 来去除前缀
        m_payload = [hex(int(m_payload_bin_str[i:i + 8], 2))[2:].zfill(2) for i in range(0, len(m_payload_bin_str), 8)]

        return m_payload
