# -*- coding: utf-8 -*-
#!/usr/bin/python3

import math
import time
import json
import redis
import threading

from sklearn.externals import joblib

listlock = threading.Lock()
bloodlist = []
imulist = []

def redisBloodSubHandle(message):

    subdata = json.loads(message['data'].decode())
    subdata["time"] = time.time()
    
#     print("blooddata",subdata["pulserate"], subdata["diastolicbp"], subdata["systolicbp"])
#     print("bloodtype", type(subdata["pulserate"]))
    # print("blooddata")

    with listlock:
        if len(imulist) > 0:
            bloodlist.append(subdata)

    return True

def redisImuSubHandle(message):
    
    subdata = json.loads(message['data'].decode())
    subdata["time"] = time.time()

#     print("imudata", subdata["acceleration"], subdata["euler"], subdata["quaternion"])
#     print("imutype", type(subdata["acceleration"]))
    # print("imudata")

    with listlock:
        imulist.append(subdata)

    return True

def main():

    redishandle = redis.StrictRedis(host="127.0.0.1", port=6379)

    bloodsub = redishandle.pubsub()
    bloodsub.subscribe(**{'blooddata': redisBloodSubHandle})
    bloodsubthread = bloodsub.run_in_thread(sleep_time=0.1)

    imusub = redishandle.pubsub()
    imusub.subscribe(**{'imudata': redisImuSubHandle})
    imusubthread = imusub.run_in_thread(sleep_time=0.1)

    scaler = joblib.load("model/scaler.pkl")
    pt_model = joblib.load("model/pt.m")
    sbp_model = joblib.load("model/sbp.m")
    dbp_model = joblib.load("model/dbp.m")

    while(True):
        global bloodlist
        global imulist
        blisttmp = []
        ilisttmp = []
        with listlock:
            blisttmp = bloodlist
            ilisttmp = imulist
        
        if (len(blisttmp) > 60) and (len(ilisttmp) > 60):
            print("size:", len(blisttmp), len(ilisttmp))
            with listlock:
                bloodlist = bloodlist[-55:]
                imulist = imulist[-55:]
            
            blistleft = 0
            blistright = len(blisttmp)
            ilistleft = 0
            ilistright = len(ilisttmp)

            alignlist = []
            while (blistleft < blistright) and (ilistleft < ilistright):
                bloodtime = blisttmp[blistleft]["time"]
                imutime = ilisttmp[ilistleft]["time"]

                if abs(bloodtime - imutime) < 1.0:
                    
                    if (blisttmp[blistleft]["pulserate"]!=255) and (blisttmp[blistleft]["pulserate"]!=0):
                        alignlist.append({
                            "time":bloodtime,
                            "pulserate":blisttmp[blistleft]["pulserate"],
                            "diastolicbp":blisttmp[blistleft]["diastolicbp"],
                            "systolicbp":blisttmp[blistleft]["systolicbp"],
                            "acceleration":ilisttmp[ilistleft]["acceleration"],
                            "quaternion":ilisttmp[ilistleft]["quaternion"]
                        })

                    blistleft += 1
                    ilistleft += 1
                elif bloodtime < imutime:
                    blistleft += 1
                else:
                    ilistleft += 1
            
            print("alignlist size:", len(alignlist))
            # print("alignlist data:", alignlist)

            if len(alignlist) >= 60:
                test_x = []
                test_unit = []
                prequaternion = None
                for elem in alignlist[-60:]:
                    test_unit.extend([elem["pulserate"], elem["systolicbp"], elem["diastolicbp"]])
                    test_unit.append(math.sqrt(pow(elem["acceleration"][0], 2) + pow(elem["acceleration"][1], 2) + pow(elem["acceleration"][2], 2))/3)
                    if prequaternion == None:
                        test_unit.append(0)
                    else:
                        pdot = prequaternion[0]*elem["quaternion"][0] + prequaternion[1]*elem["quaternion"][1] + prequaternion[2]*elem["quaternion"][2] + prequaternion[3]*elem["quaternion"][3]
                        test_unit.append(math.degrees(math.acos(round(pdot, 6))))
                    prequaternion = elem["quaternion"]
                test_x.append(test_unit)
                test_x_fit = scaler.transform(test_x)
                # print(test_x_fit)

                predblooddata = {}
                predblooddata["pulserate"] = int(pt_model.predict(test_x_fit)[0])
                predblooddata["systolicbp"] = int(sbp_model.predict(test_x_fit)[0])
                predblooddata["diastolicbp"] = int(dbp_model.predict(test_x_fit)[0])
                print(predblooddata)

                redishandle.publish("predblooddata", json.dumps(predblooddata))
        else:
            print("size:", len(blisttmp), len(ilisttmp))
        
        time.sleep(5.0)


if __name__ == "__main__":
    main()