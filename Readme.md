   # Pre-requisites:     tensorflow tf2onnx TensorRT
    git clone --recursive git@github.com:AndreV84/capstone.git
    cd capstone
   # Testing original model
    python3 test_frozen_model_TF1-2.py
   # Converting the model
    time python -m tf2onnx.convert --input frozen_graph.pb --output model.onnx --opset 12 --inputs x_in:0 --outputs decoder/mul_1:0,decoder/Softmax:0
    time /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine
   
   # Testing resulting model
    python trt_classificator_av.py --model=model.engine --image=testimage.jpg 


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

    
