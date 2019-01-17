# open redis server
gnome-terminal -x bash -c "cd ./redis-5.0.3; ./src/redis-server ./redis.conf"

sleep 1

# connect aliyun iot
gnome-terminal -x bash -c "node src/link/link.js"

sleep 1

# read blood data
gnome-terminal -x bash -c "python3 src/blood/blood.py"

sleep 1

# read imu data
gnome-terminal -x bash -c "cd src/imu/build/; cmake ..; make; ./imu"