var aliyunIot = require('aliyun-iot-device-sdk');

var device = aliyunIot.device({
    productKey: 'a1SwQ5EKSxN',
    deviceName: 'MTZRlji2GehHtZWHFu6W',
    deviceSecret: 'y2zVjni33PtSjELt5ARgWZ5HExEIO45v'
});

device.on('connect', () => {
    console.log('server connect successfully!');
});

device.on('error', (err) => {
    console.log(err)
});

device.subscribe('  /sys/a1SwQ5EKSxN/MTZRlji2GehHtZWHFu6W/thing/event/property/post/post_reply');

device.on('message', function (topic, payload) {
    console.log(topic, payload.toString());
});

setInterval(() => {
    device.postProps({
        SystolicBloodPressure: 90,
        DiastolicBloodPressure: 110
    });
}, 1000);

