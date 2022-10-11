import sys
import cv2
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

class ImagePreProcessor():

    def __init__(self, shape, dtype, preprocessor="fixed_shape_resizer"):
        """
        shape: shape of the input tensor. inputs[0]['shape']
        dtype: dtype of the input tensor as numpy. inputs[0]['dtype']
        preprocessor: aspect ratio type for resize function
        """
        if preprocessor != "fixed_shape_resizer" and preprocessor != "keep_aspect_ratio_resizer":
            print(f"Preprocessing method {preprocessor} not supported. Use fixed_shape_resizer.")
            preprocessor = "fixed_shape_resizer"

        self.preprocessor = preprocessor
        ####################
        # check_model_input_shape
        ####################
        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        # assert len(self.shape) == 4 
        if len(self.shape) == 4:
            self.batch_size = shape[0]
            assert self.batch_size > 0
            self.format = None
            self.width = -1
            self.height = -1
            if self.shape[1] == 3:
                self.format = "NCHW"
                self.height = self.shape[2]
                self.width = self.shape[3]
            elif self.shape[3] == 3:
                self.format = "NHWC"
                self.height = self.shape[1]
                self.width = self.shape[2]
            assert all([self.format, self.width > 0, self.height > 0])
        elif len(self.shape) == 3: # no batch space. only one frame tuned model.
            self.format = None
            self.width = -1
            self.height = -1
            if self.shape[0] == 3:
                self.format = "NCHW"
                self.height = self.shape[1]
                self.width = self.shape[2]
            elif self.shape[2] == 3:
                self.format = "NHWC"
                self.height = self.shape[0]
                self.width = self.shape[1]
            assert all([self.format, self.width > 0, self.height > 0])


    def preprocess(self, image):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * Resizes and pads the image to fit the input size.
        :param image: OpenCV RGB image.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def googlenet_preprocess(image):
            """
            https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet
            input:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            """
            image = np.array(image, dtype=float) * 1.0
            image[:, :, 0] -= 123.68
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 103.939
            return image


        def normalize(image):
            """
            Normalize OpenCV RGB images
            input:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            """
            miu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = np.array(image, dtype=float) / 255.
            r = (img_np[:, :, 0] - miu[0]) / std[0]
            g = (img_np[:, :, 1] - miu[1]) / std[1]
            b = (img_np[:, :, 2] - miu[2]) / std[2]
            img_np_t = np.array([r, g, b])
            img_np_t = np.transpose(img_np_t, (1, 2, 0))
            return img_np_t


        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: OpenCV RGB image.
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """

            # Get characteristics.
            height, width, _ = image.shape
            width_scale = width / self.width
            height_scale = height / self.height

            # Depending on preprocessor, box scaling will be slightly different.
            if self.preprocessor == "fixed_shape_resizer":
                scale = [self.width / width, self.height / height]
                image = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
                return image, scale
            elif self.preprocessor == "keep_aspect_ratio_resizer":
                scale = 1.0 / max(width_scale, height_scale)
                image = cv2.resize(image, (round(height * scale), round(width * scale)), interpolation=cv2.INTER_LINEAR)
                ### TODO rewrite with cv for speed up
                pad = Image.new("RGB", (self.width, self.height))
                pad.paste(pad_color, [0, 0, self.width, self.height])
                pad.paste(image, (0,0))
                return pad, scale

        scale = None

        #Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
        image, scale = resize_pad(image, (124, 116, 104))
        #image = normalize(image)
        image = googlenet_preprocess(image)
        image = np.asarray(image, dtype=self.dtype)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image, scale


class TensorRTInfer():

    def __init__(self, model_path, preprocessor="fixed_shape_resizer"):
        if preprocessor != "fixed_shape_resizer" and preprocessor != "keep_aspect_ratio_resizer":
            print(f"Preprocessing method {preprocessor} not supported. Use fixed_shape_resizer.")
            preprocessor = "fixed_shape_resizer"

        self.preprocessor = preprocessor

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        self.engine = self.load_engine(model_path)
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            # no batch model
            #if is_input:
            #    self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        #assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0


    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def destroy(self):
        if self.logger is not None:
            self.logger.destroy()
        if self.engine is not None:
            self.engine.destroy()
        if self.context is not None:
            self.context.destroy()

    def load_engine(self, model_path):
        # load tensorrt model from file
        with open(model_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        #print(f'Load model from {model_path}.')
        return engine

    def save_engine(self, engine, model_path):
        # save tensorrt model to file
        serialized_engine = engine.serialize()
        with open(model_path, "wb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            f.write(engine.serialize())
        #print(f'Save model to {model_path}.')


    def infer(self, batch, scales=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)

        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        # Process the results
        return outputs


class Classificator(object):

    def __init__(self, model_path, preprocessor="fixed_shape_resizer"):
        self.trt_model = TensorRTInfer(model_path=model_path)
        self.preprocessor = ImagePreProcessor(*self.trt_model.input_spec(), preprocessor=preprocessor)

    def execute(self, inputs):
        x = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        x, scale = self.preprocessor.preprocess(x)
        detections = self.trt_model.infer(batch=[x], scales=[scale])
        return detections
    
    def __call__(self, inputs):
        return self.execute(inputs)


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT Classification ')
    parser.add_argument("--model", default="model.engine",
        help="TensorRT model. --model=model.engine", required=True)
    parser.add_argument("--image",
        help="Input image. -image=cat.jpg", required=True)
    args = parser.parse_args()
    return args


def main(args):
    import time

    engine = args.model
    start_time=time.time()
    model = Classificator(engine)
    print(f'load: {time.time()-start_time} sec')

    # 1st frame
    start_time=time.time()
    output = model(np.zeros((224, 224, 3)).astype(np.uint8))
    print(f'1st frame: {time.time()-start_time} sec')

    import cv2
    image_path = args.image
    image = cv2.imread(image_path)
    start_time=time.time()
    output = model(image)
    print(f'infer: {time.time()-start_time} sec')
    #print(np.argmax(output))
    print(output)

if __name__ == '__main__':
    sys.exit(main(parse_args()))
