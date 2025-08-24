from socket import *
import time


def message_send(color_string):
    # 创建套接字
    tcp_clien_socket = socket(AF_INET, SOCK_STREAM)
    # 链接服务器
    server_addr = ('192.168.31.122', 7890)
    tcp_clien_socket.connect(server_addr)
    # 发送/接收数据
    send_data = color_string
    tcp_clien_socket.send(send_data.encode('utf-8'))
    rs_data = tcp_clien_socket.recv(1024)
    print(rs_data.decode('utf-8'))
    # 关闭套接字
    tcp_clien_socket.close()


def message_send_no_error(data):
    while True:
        try:
            message_send(data)
            break
        except:
            # print('error')
            time.sleep(2)

if __name__ == '__main__':
    message_send_no_error('red')