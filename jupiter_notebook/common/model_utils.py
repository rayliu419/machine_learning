
def model_parameters_number(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total parameters - {}, total_trainable_params - {}".format(total_params, total_trainable_params))