#!/bin/bash

### BEGIN INIT INFO
# Provides:	myupdate
### END INIT INFO

before_reboot(){
    sudo apt-get update
    sudo apt-get upgrade -y
    
    cd && cd Downloads/opencv-4.6.0
    mkdir build && cd build
    sudo apt-get install libconberra-gtk-module -y
    sudo apt-get install libgtk2.0-dev -y
    sudo apt-get install pkg-config -y
    sudo apt-get install python-smbus -y
    sudo apt-get install python3-pip -y
    
    cmake ..
    make 
    sudo make install

    cd
    
    git clone https://github.com/swarren21/JETSON.git
    
    sudo apt-get install python3-pip
    pip3 install --upgrade pip
    pip3 install --upgrade protobuf
    pip3 install --upgrade numpy
    pip3 install scikit-build
    pip3 install testresources
    pip3 install smbus
    
    sudo reboot
}

after_reboot(){
    pip3 install opencv-contrib-python
}

if [ -f /home/mule/Documents/rebooting-for-updates ]; then
    after_reboot
    rm /home/mule/Documents/rebooting-for-updates

else
    before_reboot
    touch /home/mule/Documents/rebooting-for-updates
    sudo reboot
fi
