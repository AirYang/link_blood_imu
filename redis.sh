# go to the project directory
cd ~/Documents/code/link_blood_imu

# open redis server
gnome-terminal -x bash -c "cd ./redis-5.0.3; ./src/redis-server ./redis.conf"

sleep 1

# open redis client
gnome-terminal -x bash -c "cd ./redis-5.0.3; ./src/redis-cli -h 127.0.0.1 -p 6379"
