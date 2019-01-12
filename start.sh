# open redis server
gnome-terminal -x bash -c "cd ~/Downloads/package/redis-5.0.3; ./src/redis-server ./redis.conf"

sleep 1

# connect aliyun iot
gnome-terminal -x bash -c "cd ~/Documents/code/link_blood_imu; node index.js"

sleep 1

# read serial data
gnome-terminal -x bash -c "cd ~/Documents/code/link_blood_imu; sudo chmod 777 /dev/ttyUSB0; python3 data.py"