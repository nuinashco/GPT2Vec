

class PartialGradNorm:
    """
    A callback that computes the gradient norm of a model's parameters.
    """

    def __init__(self, model, params=None):
        """
        Args:
            model: The model to compute the gradient norm for.
            params: The parameters to compute the gradient norm for. If None, all parameters are used.
        """
        self.model = model
        self.params = params if params is not None else list(model.parameters())

    def __call__(self):
        """
        Computes the gradient norm of the model's parameters.
        """
        grad_norm = 0.0
        for param in self.params:
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5