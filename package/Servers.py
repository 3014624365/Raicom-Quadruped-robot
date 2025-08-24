from socket import *
import time
 ### 狗2接受用
 
def message_receive():
    #创建套接字,监听套接字 负责等待新客户端进行链接
    tcp_server_socket = socket(AF_INET,SOCK_STREAM)
    #绑定本地信息
    tcp_server_socket.bind(('' ,7890))
    #将套接字由主动变为被动
    tcp_server_socket.listen(128)
    print('-------------listen---------')
    #accept 等待客户端的链接 accept产生的新套接字用来为客户端服务
    client_scoket,client_addr = tcp_server_socket.accept()

    #接收客户端的信息
    recv_data=client_scoket.recv(1024)
    #向客户端发送信息
    # client_scoket.send('呵呵呵呵'.encode('utf-8'))
    #关闭套接字
    client_scoket.close()
    tcp_server_socket.close()
    return recv_data
 
def message_receive_no_error():
    while True:
        try:
            data = message_receive().decode('utf-8')
            return data
        except:
            time.sleep(1)
            continue

if __name__ == '__main__':
    message = message_receive_no_error()
    print(message)