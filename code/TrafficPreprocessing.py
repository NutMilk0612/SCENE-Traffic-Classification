import os
from nfstream import NFStreamer, NFPlugin
import csv

INTERVAL = 50  # 时间间隔
BATCH_SIZE = 500  # 每500条流保存一次

# 定义要分析的pcap文件的绝对路径
pcap_name = "/dumeijie_data/baseline_picture/split0903__output_.pcap"

# 分别存储上行和下行数据
csv_list_up = []
csv_list_down = []
email_up = []
email_down = []

class PacketBufferPlugin(NFPlugin):

    def __init__(self, time_interval_ms=50):
        super().__init__()
        self.time_interval_ms = time_interval_ms

    def on_init(self, packet, flow):
        flow.udps.start_time = 0  # 流起始时间(13位时间戳)
        flow.udps.packet_bytes_up = []  # 每周期上行数据包字节列表
        flow.udps.packet_bytes_down = []  # 每周期下行数据包字节列表
        flow.udps.x_length = 0  # 数据包字节列表长度

    def on_update(self, packet, flow):
        # 更新流起始时间
        if flow.udps.start_time == 0:
            flow.udps.start_time = packet.time
        # 计算当前数据包所在的周期下标
        temp_index = (packet.time - flow.udps.start_time) // self.time_interval_ms
        # 判断是否造成数组越界
        if temp_index + 1 > flow.udps.x_length:
            flow.udps.packet_bytes_up.extend([0] * (temp_index + 1 - flow.udps.x_length))
            flow.udps.packet_bytes_down.extend([0] * (temp_index + 1 - flow.udps.x_length))
            flow.udps.x_length = temp_index + 1
        # 收到数据包后，按照时间戳存储在缓冲区中
        if packet.direction:  # 1 表示 dst_to_src
            flow.udps.packet_bytes_down[temp_index] += packet.payload_size
        else:  # 0 表示 src_to_dst
            flow.udps.packet_bytes_up[temp_index] += packet.payload_size

def save_to_csv(up_filename, down_filename, up_data, down_data):
    with open(up_filename, 'w', newline='') as f_up, open(down_filename, 'w', newline='') as f_down:
        writer_up = csv.writer(f_up)
        writer_down = csv.writer(f_down)
        for row_up, row_down in zip(up_data, down_data):
            writer_up.writerow(row_up)
            writer_down.writerow(row_down)
    # 清空列表
    up_data.clear()
    down_data.clear()

def analyze_pcap(pcap_file):
    # 创建带有插件参数的NFStreamer实例
    streamer = NFStreamer(source=pcap_file, udps=PacketBufferPlugin(time_interval_ms=INTERVAL))

    flow_count = 0

    # 遍历所有流
    for flow in streamer:
        if flow.udps.x_length > 10:  # 流长度足够
            flow_id = flow.id
            app_category = flow.application_category_name if flow.application_category_name else "Unknown"
            print(f"Flow ID: {flow_id}, Application Category: {app_category}")
            
            # 记录flow-id, application_category_name和流的上下行字节速率
            temp_up = [flow_id, app_category, *flow.udps.packet_bytes_up]
            temp_down = [flow_id, app_category, *flow.udps.packet_bytes_down]

            # 如果应用分类是Email，单独记录
            if app_category == "Email":
                continue
                # email_up.append(temp_up)
                # email_down.append(temp_down)
            
            csv_list_up.append(temp_up)
            csv_list_down.append(temp_down)


            
            flow_count += 1

    # 保存剩余的流数据
    if csv_list_up or csv_list_down:
        save_to_csv('up_0903.csv', 'down_0903.csv', csv_list_up, csv_list_down)

    # # 保存email流数据
    # if email_up or email_down:
    #     save_to_csv('up-email.csv', 'down-email.csv', email_up, email_down)

# 处理指定的pcap文件
analyze_pcap(pcap_name)
