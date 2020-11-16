"""
model utils for different library.
"""


def torch_model_parameters_number(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total parameters - {}, total_trainable_params - {}".format(total_params, total_trainable_params))


def tensorflow_keras_models_details(model):
    model.summary()
    model.get_config()


def tensorflow_keras_models_layer_information(model, layer_index):
    print(model.layers[layer_index].input_shape)
    print(model.layers[layer_index].output_shape)

