# -*- coding: utf-8 -*-
#!/usr/bin/python3

import sys
import time
import json
import redis

def main():
    
    # 2019-03-09T20:05:14.254Z
    imutime = time.mktime(time.strptime("2019-03-09T20:05:14.254Z"[0:20],"%Y-%m-%dT%H:%M:%S.")) + float("2019-03-09T20:05:14.000Z"[19:23])

    print(imutime)

    sys.exit()

if __name__ == "__main__":
    main()
