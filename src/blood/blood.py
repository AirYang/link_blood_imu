# -*- coding: utf-8 -*-
#!/usr/bin/python3

import sys
import time
import json
import redis
import serial
import threading

status = "read"
statuslock = threading.Lock()
calibration = ""


def writeSerialPort(serialport, writedata):

    # print(writedata)

    if serialport.isOpen():
        serialport.write(bytes().fromhex(writedata))
        return True
    else:
        print("serialport is closed!")
        return False


def readSerialPortAndSend(serialport, redishandle, bytesize):
    if serialport.isOpen():
        data = serialport.read(6)

        if(len(data) < bytesize):
            print("serialport can't read bytesize!")
            return False

        else:

            if int(data.hex()[0:2], 16) == 253:
                # print("read")
                blooddata = {}
                blooddata["systolicbp"] = int(data.hex()[2:4], 16)
                blooddata["diastolicbp"] = int(data.hex()[4:6], 16)
                blooddata["pulserate"] = int(data.hex()[6:8], 16)
                redishandle.publish("blooddata", json.dumps(blooddata))
                return True

            elif int(data.hex()[0:2], 16) == 254:
                # print("calibration")
                calibrationflag = int(data.hex()[6:8], 16)
                if calibrationflag == 0:
                    print("calibration true")
                    return True
                elif calibrationflag == 2:
                    print("calibration false")
                    return False
                else:
                    print("calibration...")
                    time.sleep(0.1)
                    return readSerialPortAndSend(serialport, redishandle, bytesize)

            elif int(data.hex()[0:2], 16) == 250:
                # print("clear")
                erasedata = {}
                erasedata["erase"] = int(data.hex()[6:8], 16)
                redishandle.publish("erase", json.dumps(erasedata))
                return True

    else:
        print("serialport is closed!")
        return False


def redisSetSubHandle(message):

    subdata = json.loads(message['data'].decode())

    tmp = "FE" + ("%02X%02X%02X" % (subdata['SystolicBloodPressure'],
                                      subdata['DiastolicBloodPressure'], subdata['PulseRate'])) + "0000"

    with statuslock:
        global status
        global calibration
        status = "calibration"
        calibration = tmp

    return True


def redisClearSubHandle(message):

    subdata = json.loads(message['data'].decode())

    if subdata['Erase'] == 1:
        with statuslock:
            global status
            status = "clear"

    return True


def main():

    serialport = serial.Serial(
        "/dev/ttyUSBBLOOD", baudrate=115200, stopbits=serial.STOPBITS_ONE, timeout=0.5)

    redishandle = redis.StrictRedis(host="127.0.0.1", port=6379)

    redissetsub = redishandle.pubsub()
    redissetsub.subscribe(**{'property/set': redisSetSubHandle})
    setsubthread = redissetsub.run_in_thread(sleep_time=0.1)

    redisclearsub = redishandle.pubsub()
    redisclearsub.subscribe(**{'property/clear': redisClearSubHandle})
    clearsubthread = redisclearsub.run_in_thread(sleep_time=0.1)

    while(serialport.isOpen()):

        global status
        global calibration
        statustmp = ""
        calibrationtmp = ""
        with statuslock:
            statustmp = status
            calibrationtmp = calibration

        if statustmp == "read":

            writeSerialPort(serialport, "FD0000000000")
            readSerialPortAndSend(serialport, redishandle, 6)

        elif statustmp == 'calibration':

            writeSerialPort(serialport, calibrationtmp)
            if readSerialPortAndSend(serialport, redishandle, 6) == True:
                with statuslock:
                    status = "read"
                    calibration = ""
            else:
                with statuslock:
                    status = "clear"
                    calibration = ""

        elif statustmp == 'clear':

            writeSerialPort(serialport, "FA0000000000")
            readSerialPortAndSend(serialport, redishandle, 6)

            with statuslock:
                status = "read"

        time.sleep(2.0)

    setsubthread.stop()
    clearsubthread.stop()
    sys.exit()


if __name__ == "__main__":
    main()
