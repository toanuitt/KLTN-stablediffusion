import os
import torch
from .networks import define_G

class Pix2PixModel():
    def __init__(self, opt):
        self.rank = opt.rank
        self.device = torch.device(f"cuda:{self.rank}") if self.rank is not None else torch.device('cpu')
        
        if "expand" in opt.extra_info:
            opt.input_nc = opt.input_nc + 1
        if "label" in opt.extra_info:
            opt.input_nc = opt.input_nc + 1

        self.generator = define_G(opt.input_nc,
                                      opt.output_nc,
                                      opt.ngf,
                                      opt.netG,
                                      opt.norm,
                                      opt.dropout,
                                      output_type=opt.output_type,
                                      unet_backbone=opt.unet_backbone)
        
        assert os.path.exists(opt.generator_path), f"Cannot find {opt.generator_path}"
        if opt.generator_path is not None:
            self.load_network(opt.generator_path)


    def load_network(self, generator_path, weights_only=True):
        print('loading the model from %s' % generator_path)
        state_dict = torch.load(generator_path, map_location=str(self.device), weights_only=weights_only)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(state_dict, self.generator, key.split('.'))
        self.generator.load_state_dict(state_dict)


    def predict(self, input):
        with torch.no_grad():
            predict = self.generator(input.to(self.device))

        return predict.detach().cpu()