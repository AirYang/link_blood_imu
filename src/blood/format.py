# -*- coding: utf-8 -*-
#!/usr/bin/python3

import sys
import time
import json
import redis

def main():

    redishandle = redis.StrictRedis(host="127.0.0.1", port=6379)
    for key in redishandle.lrange("imu.blood.relation", 0, -1):
        keyobj = json.loads(key.decode())
        
        imuname = keyobj['imuname']
        imudatas = redishandle.lrange(imuname, 0, -1)[::-1]
        
        bloodname = keyobj['bloodname']
        blooddatas = redishandle.lrange(bloodname, 0, -1)[::-1]
        
        # print(bloodname, end="\n")
        # for blooddata in blooddatas:
            # print(json.loads(blooddata.decode())["time"], end="\n")
            # print(time.mktime(time.strptime(json.loads(blooddata.decode())["time"][0:20],"%Y-%m-%dT%H:%M:%S.")))
            # pass
        
        # print(imuname, end="\n")
        # for imudata in imudatas:
            # print(json.loads(imudata.decode())["time"], end="\n")
            # print(time.mktime(time.strptime(json.loads(imudata.decode())["time"][0:20],"%Y-%m-%dT%H:%M:%S.")))
            # pass
        
        paircount = 0
        imuleft = 0
        imuright = len(imudatas)
        bloodleft = 0
        bloodright = len(blooddatas)

        while (imuleft < imuright) and (bloodleft < bloodright):
            
            imudata = json.loads(imudatas[imuleft].decode())
            imutime = time.mktime(time.strptime(imudata["time"][0:20],"%Y-%m-%dT%H:%M:%S.")) + float(imudata["time"][19:23])

            blooddata = json.loads(blooddatas[bloodleft].decode())
            bloodtime = time.mktime(time.strptime(blooddata["time"][0:20],"%Y-%m-%dT%H:%M:%S.")) + float(blooddata["time"][19:23])

            if abs(imutime-bloodtime) < 1.0:
                print(imutime, bloodtime)
                print(imudata["time"], blooddata["time"])
                paircount += 1
                imuleft += 1
                bloodleft += 1

            elif imutime > bloodtime:
                bloodleft += 1
            
            else:
                imuleft += 1
        
        print(imuright, imuleft)
        print(bloodright, bloodleft)
        print(paircount)
        
        break

    sys.exit()

if __name__ == "__main__":
    main()
