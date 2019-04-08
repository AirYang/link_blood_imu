# -*- coding: utf-8 -*-
#!/usr/bin/python3

# import os
import csv
import sys
import time
import json
import redis

def main():

    redishandle = redis.StrictRedis(host="127.0.0.1", port=6379)

    fieldnames = ['group', 'time', 'pulserate', 'systolicbloodpressure', 'diastolicbloodpressure', 'quaternion', 'euler', 'acceleration']
    csvfile = open('data/sensor.csv', 'w+', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    groupcount = 65
    for key in redishandle.lrange("imu.blood.relation", 0, -1):
        
        keyobj = json.loads(key.decode())
        
        imuname = keyobj['imuname']
        imudatas = redishandle.lrange(imuname, 0, -1)[::-1]
        
        bloodname = keyobj['bloodname']
        blooddatas = redishandle.lrange(bloodname, 0, -1)[::-1]
        
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
                writer.writerow({'group':groupcount, 'time':blooddata["time"], 'pulserate':blooddata['pulserate'], 'systolicbloodpressure':blooddata['systolicbp'], 'diastolicbloodpressure':blooddata['diastolicbp'],'quaternion':imudata['quaternion'], 'euler':imudata['euler'], 'acceleration':imudata['acceleration']})

                paircount += 1
                imuleft += 1
                bloodleft += 1

            elif imutime > bloodtime:
                bloodleft += 1
            
            else:
                imuleft += 1
        
        groupcount -= 1
        print(imuname,imuright, imuleft)
        print(bloodname,bloodright, bloodleft)
        print(groupcount,paircount)
        # break

    csvfile.close()
    sys.exit()

if __name__ == "__main__":
    main()
