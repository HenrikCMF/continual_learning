from TCP_code import TCP_COM
local_IP="192.168.1.107"
PORT=5000
rec_ip="192.168.1.101"
com_obj=TCP_COM(local_IP, PORT, rec_ip, PORT)
com_obj.send_file("307.jpg")