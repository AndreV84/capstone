import tensorflow as tf

# Load frozen graph
graph_def = tf.GraphDef()
with tf.gfile.GFile("frozen_graph.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())

    print('========== nodes ==========')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node]
    for node in nodes:
        print(node)

    print('========== inputs ==========')
    input_nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in input_nodes:
        print(node)

    print('========== outputs ==========')
    name_list = []
    input_list = []
    for n in graph_def.node:
        name_list.append(n.name)
        for name in n.input:
            input_list.append(name)

    outputs = set(name_list) - set(input_list)
    output_nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.name in outputs]
    for node in output_nodes:
        print(node)
