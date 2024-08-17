def get_model_params(m, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())
