# coding: utf-8

import time  
import socket
from .utils import *
import base64
import threading

class Client(object):
    def __init__(self, SEVERIP='127.0.0.1', SERVERPORT=5123, DEBUG=True):
        self.SEVERIP = SEVERIP
        self.SERVERPORT = SERVERPORT
        self.sock = self.connect()
        self.DEBUG = DEBUG
        self.receive_buf = b''
        self.notify = []
        self.game_variable = ''
        self.objid_list = ''
        self.check_pos = ''
        self.others = []

    def connect(self, ):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM,) 
        self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.SEVERIP, self.SERVERPORT))
        self.t1 = threading.Thread(target=self.receive,)
        self.t1.start()
        return self.sock

    def send(self, cmd, ):
        try:
            self.sock.sendall(base64.b64encode(bytes(cmd, encoding='utf8')) + b'\n')
            if self.DEBUG:
                print('send over:%s'%cmd)
        except:
            if self.DEBUG:
                print('send failed!!!!!!!!!!!!!!!!!!')

    
    def receive(self, ):
        while True:
            st = self.sock.recv(81920)
            self.receive_buf += st
            sp = self.receive_buf.split(b'\n')
            if len(sp) > 1:
                for i in range(len(sp) - 1):
                    st = base64.b64decode(sp[i]).decode()
                    if st[-1] == '`':
                        st = st[:-1]
                    if self.DEBUG:
                        print('receive:', st)
                    if st.find('notify')>-1:
                        self.notify.append(st)
                    elif st.find('game_variable')>-1:
                        self.game_variable = '' if st.find('`') < 0 else st
                    elif st.find('objid_list') > -1:
                        self.objid_list = '' if st.find('`') < 0 else st
                    else:
                        self.others.append(st)
                self.receive_buf = sp[-1]