  | Sample GUI      | Vehicle      |  Cleanliness map
|------------|-------------|------------------|
| <img src="https://github.com/AndreV84/capstone/blob/main/misc/leaves.png" width="250"> | <img src="https://github.com/AndreV84/capstone/blob/main/misc/truck_annotated.png" width="250"> | <img src="https://github.com/AndreV84/capstone/blob/main/misc/map.png" width="250">
  
   
   
   # Pre-requisites:     tensorflow tf2onnx TensorRT
    git clone --recursive git@github.com:AndreV84/capstone.git
    cd capstone
   # Testing original model
    cd testing_model_code/from_model_supplier
    cp ../../inputs/frozen_graph.pb .
    python3 test_frozen_model_TF1-2.py
   # Converting the model
   | Tensorflow ---->      | ONNX      |  
|------------|-------------|
| <img src="https://github.com/AndreV84/capstone/blob/main/misc/TF.png" width="250"> | <img src="https://github.com/AndreV84/capstone/blob/main/misc/onnx.png" width="250">
   
    cp inputs/frozen_graph.pb .
    time python3 -m tf2onnx.convert --input frozen_graph.pb --output model.onnx --opset 12 --inputs x_in:0 --outputs decoder/mul_1:0,decoder/Softmax:0
    time /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine
    
    
   | ONNX     | TensorRT   |  
|------------|-------------|
| <img src="https://github.com/AndreV84/capstone/blob/main/misc/onnx_runtime_nv.png" width="250"> | <img src="https://github.com/AndreV84/capstone/blob/main/misc/Tensorrt.png" width="250">
   
   
   # Testing resulting model
    cd outputs
    cp ../testing_model_code/naisy/trt_classificator_av.py.
    python3 trt_classificator_av.py --model=model.engine --image=testimage.jpg 


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
   # Further acceleration
    
    #FP16
    time /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

    
    #INT8
    time /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8
    
    # infer
    python trt_classificator_av.py --model=model_fp32.engine --image=testimage.jpg
    python trt_classificator_av.py --model=model_fp16.engine --image=testimage.jpg
    python trt_classificator_av.py --model=model_int8.engine --image=testimage.jpg


    
   # Even further acceleration 
    
    FP16 & INT8 with Deepstream [ up to 44 fps with INT8, 27 fps with FP16]

    
