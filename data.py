# -*- coding: utf-8 -*-
#!/usr/bin/python3

import sys
import time
import redis
import serial


def writeSerialPort(serialport):
    if serialport.isOpen():
        serialport.write(bytes().fromhex("FD0000000000"))
    else:
        print("serialport is closed!")
        sys.exit()


def readSerialPortAndSend(serialport, redishandle, bytesize):
    if serialport.isOpen():
        data = serialport.read(6)
        if(len(data) < bytesize):
            return print("serialport can't read bytesize")
        else:
            redishandle.publish("serialdata",data)
            return print(data)
    else:
        print("serialport is closed!")
        sys.exit()


def main():
    serialport = serial.Serial(
        "/dev/ttyS0", baudrate=115200, stopbits=serial.STOPBITS_ONE, timeout=0.5)
    redishandle = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)

    bytesize = 6
    while(serialport.isOpen()):
        writeSerialPort(serialport)
        readSerialPortAndSend(serialport, redishandle, bytesize)
        time.sleep(0.1)
    sys.exit()


if __name__ == "__main__":
    main()
