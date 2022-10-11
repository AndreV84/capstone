import sys
import tensorflow as tf
if float(tf.__version__[0]) > 1:
    print('Import TF2 as TF1', file=sys.stderr)
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    tf.disable_v2_behavior()

import numpy as np
import time
import cv2
import glob


# function that creates random images
def rand_image_TF(image_size, series_size):
    if series_size > 0:
        image = np.random.randint(255, size=(series_size,image_size[0],image_size[1],3), dtype=np.uint8)
    else:
        image = np.random.randint(255, size=(image_size[0],image_size[1],3), dtype=np.uint8)
    return image

image_size = [1056,1920]

# We load the protobuf file from the disk and parse it to retrieve the 
# unserialized graph_def
frozen_graph_filename = "frozen_graph.pb"

with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Then, we import the graph_def into a new Graph and returns it 
with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="")

sess = tf.Session(graph=graph)


# get / create image
image = rand_image_TF(image_size,0)

# image_names = glob.glob("test_images/*.jpg")
# image = cv2.imread(image_names[0])
# image = cv2.resize(image,(image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC)

# print('\n')
# print(len(graph.get_operations()))
# print('\n')
# print(graph.get_operations())
# print('\n')

x_in = graph.get_tensor_by_name("x_in:0")
pred_boxes = graph.get_tensor_by_name("decoder/mul_1:0")
pred_confidences = graph.get_tensor_by_name("decoder/Softmax:0")

feed = {x_in: image}

for i in range(10):
    start = time.time()
    (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
    end = time.time()
    #display elapsed time for the inference
    elaps_time = (end-start)
    print(elaps_time)
print(np_pred_boxes)
print(np_pred_confidences)
