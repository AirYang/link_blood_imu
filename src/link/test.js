let testtime = new Date();
console.log(testtime);
console.log(testtime.toUTCString());

let beijingtime = new Date( testtime.getTime() - testtime.getTimezoneOffset() * 60 * 1000 );
console.log(beijingtime);
console.log(beijingtime.toUTCString());


console.log(beijingtime.toJSON());