var aliyunIot = require('aliyun-iot-device-sdk');

var device = aliyunIot.device({
    productKey: 'a1SwQ5EKSxN',
    deviceName: 'MTZRlji2GehHtZWHFu6W',
    deviceSecret: 'y2zVjni33PtSjELt5ARgWZ5HExEIO45v'
});

var redis = require('redis');

var subscriber = redis.createClient();

device.on('connect', () => {
    console.log('aliyun iot connect...');
});

device.on('error', (error) => {
    console.log(err)
});

device.subscribe('/sys/a1SwQ5EKSxN/MTZRlji2GehHtZWHFu6W/thing/event/property/post/post_reply');

device.on('message', function (topic, payload) {
    console.log(topic, payload.toString());
});

subscriber.on("connect", function (error) {
    console.log("redis connect...");
});

subscriber.on("error", function (error) {
    console.log("redis error:", error);
});

subscriber.on("message", function (channel, message) {

    // recv blood data
    if (channel == "blooddata") {
        let buffer = Buffer.from(message, "hex");
        if (buffer.readUInt8(0) == 253) {
            console.log(channel, buffer.readUInt8(1), buffer.readUInt8(2));
            device.postProps({
                SystolicBloodPressure: buffer[1],
                DiastolicBloodPressure: buffer[2]
            });
        }
    }

    //recv imu data
    else if (channel == "imudata") {

    }
});

subscriber.subscribe("blooddata");
subscriber.subscribe("imudata");
