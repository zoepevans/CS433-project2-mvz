from NN import *

feature_choice = "2-body"  # Options: "2-body", "3-body", "4-body", "2+3-body", "2+3+4-body"
dataset = MyDataset(feature_choice)

#uncomment only the NN you want

SimpleNN_1_layer_graph(dataset)
#SimpleNN_multiple_layers_graph(dataset)
#SimpleNN_multiple_layers_stopping_criteria_graph(dataset)
#SimpleNN_one_layer_multiple_nodes_graph(dataset)