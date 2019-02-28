// var aliyunIot = require('aliyun-iot-device-sdk');
// var device = aliyunIot.device({
//     productKey: 'a1SwQ5EKSxN',
//     deviceName: 'MTZRlji2GehHtZWHFu6W',
//     deviceSecret: 'y2zVjni33PtSjELt5ARgWZ5HExEIO45v'
// });

// var redis = require('redis');
// var sub = redis.createClient();
// var set = redis.createClient();

var Redis = require('ioredis');
var sub = new Redis();
var set = new Redis();

var bloodname = "";
var imuname = "";

// device.on('connect', () => {
//     console.log('aliyun iot connect...');
// });

// device.on('error', (error) => {
//     console.log(error);
// });

// device.on('message', function (topic, payload) {
//     // console.log(topic, payload.toString());
// });

// device.serve('property/set', function (params) {

//     if (typeof (params.Erase) != "undefined") {
//         pub.publish("property/clear", JSON.stringify(params));
//     }

//     else if ((typeof (params.SystolicBloodPressure) == "undefined") || (typeof (params.DiastolicBloodPressure) == "undefined") || (typeof (params.PulseRate) == "undefined")) {
//         console.log("property/set lack of parameters");
//     }

//     else {
//         pub.publish("property/set", JSON.stringify(params));
//     }
// });

set.on("connect", function (error) {
    console.log("redis set connect...");
});

set.on("error", function (error) {
    console.log("redis set error:", error);
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

        if (bloodname == "") {
            bloodname = "blooddata." + bloodtime.getTime();
            set.sadd("blooddata.name", bloodname);
        }

        blooddata.time = bloodtime.getTime();
        set.lpush(bloodname, JSON.stringify(blooddata));

        // if ((blooddata.pulserate == 0) || (blooddata.systolicbp == 0) || (blooddata.diastolicbp == 0) || (blooddata.pulserate == 255) || (blooddata.systolicbp == 255) || (blooddata.diastolicbp == 255)) {
        //     device.postProps({
        //         Status: 5
        //     });
        // }

        // else {
        //     device.postProps({
        //         Status: 0
        //     });
        // }
    }

    // recv imu data
    else if (channel == "imudata") {
        let imudata = JSON.parse(message);
        let imutime = new Date();
        // console.log("imutime", imutime.getTime());
        // console.log("euler", imudata.euler);
        // console.log("quaternion", imudata.quaternion);
        // console.log("acceleration", imudata.acceleration);

        if (imuname == "") {
            imuname = "imudata." + imutime.getTime();
            set.sadd("imudata.name", imuname);

            if (bloodname == "") {
                console.error("bloodname is empty!");
                process.exit();
            }
            
            let relation = {};
            relation['imuname'] = imuname;
            relation['bloodname'] = bloodname;
            set.lpush("imu.blood.relation", JSON.stringify(relation));
        }

        imudata.time = imutime.getTime();
        set.lpush(imuname, JSON.stringify(imudata));
    }

    // recv erase status
    // else if (channel == "erase") {
    //     let status = JSON.parse(message);
    //     console.log("erase", status.erase);

    //     // if (status.erase == 1) {
    //     //     device.postProps({
    //     //         Status: 1
    //     //     });
    //     // }
    // }

    //recv calibration status
    // else if (channel == "calibration") {
    //     let status = JSON.parse(message);
    //     console.log("calibration", status.calibration);

    //     // if (status.calibration == "success") {
    //     //     device.postProps({
    //     //         Status: 2
    //     //     });
    //     // }

    //     // else if (status.calibration == "failure") {
    //     //     device.postProps({
    //     //         Status: 3
    //     //     });
    //     // }

    //     // else if (status.calibration == "processing") {
    //     //     device.postProps({
    //     //         Status: 4
    //     //     });
    //     // }
    // }
});

// sub.subscribe("erase");
sub.subscribe("imudata");
sub.subscribe("blooddata");
// sub.subscribe("calibration")

// device.subscribe('/sys/a1SwQ5EKSxN/MTZRlji2GehHtZWHFu6W/thing/event/property/post/post_reply');