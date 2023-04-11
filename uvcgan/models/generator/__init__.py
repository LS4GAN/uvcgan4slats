from uvcgan.base.networks    import select_base_generator
from uvcgan.base.weight_init import init_weights
from uvcgan.torch.funcs      import prepare_model

from .vit       import ViTGenerator
from .vitunet   import ViTUNetGenerator

def select_generator(name, **kwargs):
    if name == 'vit-v0':
        return ViTGenerator(**kwargs)

    if name == 'vit-unet':
        return ViTUNetGenerator(**kwargs)

    input_shape  = kwargs.pop('input_shape')
    output_shape = kwargs.pop('output_shape')

    assert input_shape == output_shape
    return select_base_generator(name, image_shape = input_shape, **kwargs)

def construct_generator(model_config, input_shape, output_shape, device):
    model = select_generator(
        model_config.model,
        input_shape  = input_shape,
        output_shape = output_shape,
        **model_config.model_args
    )

    model = prepare_model(model, device)
    init_weights(model, model_config.weight_init)

    return model

