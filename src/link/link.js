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
    // console.log(topic, payload.toString());
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
        let bloodtime = new Date();
        // console.log("bloodtime", bloodtime.getTime());
        // console.log("pulserate", blooddata.pulserate);
        // console.log("systolicbp", blooddata.systolicbp);
        // console.log("diastolicbp", blooddata.diastolicbp);
        // console.log("blooddata", blooddata);

        if ((blooddata.pulserate == 0) || (blooddata.systolicbp == 0) || (blooddata.diastolicbp == 0) || (blooddata.pulserate == 255) || (blooddata.systolicbp == 255) || (blooddata.diastolicbp == 255)) {
            device.postProps({
                Status: 5
            });
        }

        else {
            device.postProps({
                Status: 0
            });
        }
    }

    // recv imu data
    else if (channel == "imudata") {
        let imudata = JSON.parse(message);
        let imutime = new Date();
        // console.log("imutime", imutime.getTime());
        // console.log("euler", imudata.euler);
        // console.log("quaternion", imudata.quaternion);
        // console.log("acceleration", imudata.acceleration);
        // console.log("imudata", imudata);
    }

    // recv erase status
    else if (channel == "erase") {
        let status = JSON.parse(message);
        console.log("erase", status.erase);

        if (status.erase == 1) {
            device.postProps({
                Status: 1
            });
        }
    }

    //recv calibration status
    else if (channel == "calibration") {
        let status = JSON.parse(message);
        console.log("calibration", status.calibration);

        if (status.calibration == "success") {
            device.postProps({
                Status: 2
            });
        }

        else if (status.calibration == "failure") {
            device.postProps({
                Status: 3
            });
        }

        else if (status.calibration == "processing") {
            device.postProps({
                Status: 4
            });
        }
    }

    //recv predict blooddata
    else if (channel == "predblooddata") {
        let blooddata = JSON.parse(message);
        // console.log("predblooddata", blooddata);
        console.log("predblooddata:", blooddata.pulserate, blooddata.diastolicbp, blooddata.systolicbp)

        device.postProps({
            PulseRate: blooddata.pulserate,
            DiastolicBloodPressure: blooddata.diastolicbp,
            SystolicBloodPressure: blooddata.systolicbp
        })
    }
});

sub.subscribe("erase");
sub.subscribe("imudata");
sub.subscribe("blooddata");
sub.subscribe("calibration")
sub.subscribe("predblooddata")

device.subscribe('/sys/a1SwQ5EKSxN/MTZRlji2GehHtZWHFu6W/thing/event/property/post/post_reply');