# -*- coding: utf-8 -*-
#!/usr/bin/python3

import csv
import sys
import time
import json
import redis

def main():
    
    # 2019-03-09T20:05:14.254Z
    # imutime = time.mktime(time.strptime("2019-03-09T20:05:14.254Z"[0:20],"%Y-%m-%dT%H:%M:%S.")) + float("2019-03-09T20:05:14.000Z"[19:23])

    # print(imutime)

    # with open('data/cuff.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['group', 'time', 'pulserate', 'systolicbloodpressure', 'diastolicbloodpressure']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     writer.writerow({'group':20, 'time':"2019-03-09T19:54:00.000Z", 'pulserate':90, 'systolicbloodpressure':104, 'diastolicbloodpressure':63})

    with open('data/person.csv', 'w+', newline='') as csvfile:
        fieldnames = ['group', 'age', 'weight', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for index in range(1,21)[::-1]:
            writer.writerow({'group':index, 'age':22, 'weight':67, 'height':1.73})

    sys.exit()

if __name__ == "__main__":
    main()
