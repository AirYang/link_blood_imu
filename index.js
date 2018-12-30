var aliyunIot = require('aliyun-iot-device-sdk');

var device = aliyunIot.device({
    productKey: 'a1SwQ5EKSxN',
    deviceName: 'MTZRlji2GehHtZWHFu6W',
    deviceSecret: 'y2zVjni33PtSjELt5ARgWZ5HExEIO45v'
});

var redis = require('redis');

var subscriber = redis.createClient({ db: 0 });

device.on('connect', () => {
    console.log('aliyunIot connect...');
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
    console.log(Buffer.from(message, "binary"), channel);

    device.postProps({
        SystolicBloodPressure: 90,
        DiastolicBloodPressure: 110
    });
});

subscriber.subscribe("serialdata");

