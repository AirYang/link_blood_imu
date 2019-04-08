#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include "LpmsSensorI.h"
#include "LpmsSensorManagerI.h"
#endif
#ifdef __GNUC__
#include "lpsensor/LpmsSensorI.h"
#include "lpsensor/LpmsSensorManagerI.h"
#endif

extern "C"{
#include "hiredis/hiredis.h"
}

#include "json.hpp"
using json = nlohmann::json;

int main(int argc, char *argv[])
{
    ImuData d;

    // Gets a LpmsSensorManager instance
    LpmsSensorManagerI* manager = LpmsSensorManagerFactory();
    // 
    // DEVICE_LPMS_B        LPMS-B (Bluetooth)
    // DEVICE_LPMS_U        LPMS-CU / LPMS-USBAL (USB)
    // DEVICE_LPMS_C        LPMS-CU / LPMS-CANAL(CAN bus)
    // DEVICE_LPMS_BLE      LPMS-BLE (Bluetooth low energy)
    // DEVICE_LPMS_RS232    LPMS-UARTAL (RS-232)
    // DEVICE_LPMS_B2       LPMS-B2
    // DEVICE_LPMS_U2       LPMS-CU2/URS2/UTTL2/USBAL2 (USB)
    // DEVICE_LPMS_C2       LPMS-CU2/CANAL2 (CAN)

    // Connects to LPMS-B2 sensor with address 00:11:22:33:44:55 
    //LpmsSensorI* lpms = manager->addSensor(DEVICE_LPMS_B2, "00:11:22:33:44:55");
    // Connects to LPMS-CURS2 sensor try virtual com port 
    LpmsSensorI* lpms = manager->addSensor(DEVICE_LPMS_RS232, "/dev/ttyUSBIMU");

    // Connect redis server
    // struct timeval timeout = { 1, 500000 };
    redisContext *c = redisConnectWithTimeout("127.0.0.1", 6379, { 1, 500000 });
    if (c == NULL || c->err) {
        if (c) {
            printf("Redis connection error: %s\n", c->errstr);
            redisFree(c);
        } else {
            printf("Redis connection error: can't allocate redis context\n");
        }
        exit(1);
    }

    // double v[3] = {0};
    while(1) 
    {		 
        // Checks, if sensor is connected
        if (lpms->getConnectionStatus() == SENSOR_CONNECTION_CONNECTED &&
            lpms->hasImuData()) 
        {
            // Reads quaternion data
            d = lpms->getCurrentData();

            // Shows data
            // printf("---------------   timestamp=%8.2f   ---------------\n", d.timeStamp);
            // printf("cal acc sen [%8.2f, %8.2f, %8.2f]\n", d.a[0], d.a[1], d.a[2]);
            // printf("cal gyr sen [%8.2f, %8.2f, %8.2f]\n", d.g[0], d.g[1], d.g[2]);
            // printf("cal mag sen [%8.2f, %8.2f, %8.2f]\n", d.b[0], d.b[1], d.b[2]);
            // printf("quaternion  [%8.2f, %8.2f, %8.2f, %8.2f]\n", d.q[0], d.q[1], d.q[2], d.q[3]);
            // printf("euler       [%8.2f, %8.2f, %8.2f]\n", d.r[0], d.r[1], d.r[2]);
            // printf("lin acc     [%8.2f, %8.2f, %8.2f]\n", d.linAcc[0], d.linAcc[1], d.linAcc[2]);

            json imudata_json;
            imudata_json["euler"] = {d.r[0], d.r[1], d.r[2]};
            imudata_json["quaternion"] = {d.q[0], d.q[1], d.q[2], d.q[3]};
            imudata_json["acceleration"] = {d.linAcc[0], d.linAcc[1], d.linAcc[2]};
            printf("%s\n", imudata_json.dump().c_str());

            // compute speed
            // v[0] += d.linAcc[0] * 0.01;
            // v[1] += d.linAcc[1] * 0.01;
            // v[2] += d.linAcc[2] * 0.01;
            // printf("%lf\n", sqrt(pow(v[0], 2)+pow(v[1], 2)+pow(v[2], 2)));

            redisReply *pub_reply = static_cast<redisReply *>(redisCommand(c, "PUBLISH %b %b", "imudata", (size_t)7, imudata_json.dump().c_str(), imudata_json.dump().size()));
            // printf("redis publish return [%s]\n", pub_reply->str);
            freeReplyObject(pub_reply);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // Removes the initialized sensor
    manager->removeSensor(lpms);

    // Deletes LpmsSensorManager object 
    delete manager;

     /* Disconnects and frees the context */
    redisFree(c);

    return 0;
}

