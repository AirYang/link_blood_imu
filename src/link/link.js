var aliyunIot = require('aliyun-iot-device-sdk');
var device = aliyunIot.device({
    productKey: 'a1SwQ5EKSxN',
    deviceName: 'MTZRlji2GehHtZWHFu6W',
    deviceSecret: 'y2zVjni33PtSjELt5ARgWZ5HExEIO45v'
});

var redis = require('redis');
var sub = redis.createClient();
var pub = redis.createClient();

device.on('connect', () => {
    console.log('aliyun iot connect...');
});

device.on('error', (error) => {
    console.log(error);
});

device.on('message', function (topic, payload) {
    console.log(topic, payload.toString());
});

device.serve('property/set', function (params) {

    if (typeof (params.Erase) != "undefined") {
        pub.publish("property/clear", JSON.stringify(params));
    }

    else if ((typeof (params.SystolicBloodPressure) == "undefined") || (typeof (params.DiastolicBloodPressure) == "undefined") || (typeof (params.PulseRate) == "undefined")) {
        console.log("property/set lack of parameters");
    }

    else {
        pub.publish("property/set", JSON.stringify(params));
    }
});

pub.on("connect", function (error) {
    console.log("redis pub connect...");
});

pub.on("error", function (error) {
    console.log("redis pub error:", error);
});

sub.on("connect", function (error) {
    console.log("redis sub connect...");
});

sub.on("error", function (error) {
    console.log("redis sub error:", error);
});

sub.on("message", function (channel, message) {

    // recv blood data
    if (channel == "blooddata") {
        let blooddata = JSON.parse(message);
        console.log("systolicbp", blooddata.systolicbp);
        console.log("diastolicbp", blooddata.diastolicbp);
        console.log("pulserate", blooddata.pulserate);
    }

    // recv imu data
    else if (channel == "imudata") {
        let imudata = JSON.parse(message);
        console.log("quaternion", imudata.quaternion);
        console.log("euler", imudata.euler);
    }

    // recv erase data
    else if (channel == "erase") {
        let erasedata = JSON.parse(message);
        console.log("erase", erasedata.erase);
    }
});

sub.subscribe("erase");
sub.subscribe("imudata");
sub.subscribe("blooddata");

device.subscribe('/sys/a1SwQ5EKSxN/MTZRlji2GehHtZWHFu6W/thing/event/property/post/post_reply');