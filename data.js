const PortName = '/dev/ttyUSB0';

const SerialPort = require('serialport');

const port = new SerialPort(
    PortName, {
        bandRate: 115200
    }
);

port.on('data', (data) => {
    console.log('uart recv:', data);
});

port.on('open', () => {
    console.log(PortName, 'open successfully!');
});

setInterval(() => {
    if (port.isOpen) {
        port.write('FD0000000000', 'hex', (err) => {
            if (err) {
                return console.log(PortName, 'write error:', err.message);
            }
            console.log(PortName, 'message written');
        });
    }
}, 1000);





