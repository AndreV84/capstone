    git clone --recursive git@github.com:AndreV84/capstone.git
    cd capstone
    
   # Pipeline interface
    cd docker
    sudo su
    #Tensorflow 2.7.0
    ./run-jetson-jp461-base.sh
     #Tensorflow 1.15.5
     ./run-jetson-jp461-donkeycar-overdrive3.sh
      #JupyterLab
      http://jetson_ip_address:8888
      pass: jupyter

    
