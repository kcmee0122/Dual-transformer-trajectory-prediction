def Count_parameters(Model):
    # print('Model:', Model._get_name(), '/ Trainable params:', sum(p.numel() for p in Model.parameters() if p.requires_grad))
    return sum(p.numel() for p in Model.parameters() if p.requires_grad)