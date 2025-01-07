import os
import torch
from .networks import define_G


class Pix2PixModel:
    def __init__(self, opt):
        self.rank = opt["rank"]
        self.device = (
            torch.device(f"cuda:{self.rank}")
            if self.rank is not None
            else torch.device("cpu")
        )

        self.input_nc = opt["input_nc"]
        if "expand" in opt["extra_info"]:
            self.input_nc += 1
        if "label" in opt["extra_info"]:
            self.input_nc += 1

        self.generator = define_G(
            self.input_nc,
            opt["output_nc"],
            opt["ngf"],
            opt["netG"],
            opt["norm"],
            opt["dropout"],
            output_type=opt["output_type"],
            unet_backbone=opt["unet_backbone"],
        )

        assert os.path.exists(
            opt["generator_path"]
        ), f"Cannot find {opt.generator_path}"
        if opt["generator_path"] is not None:
            self.load_network(opt["generator_path"])

    def load_network(self, generator_path, weights_only=True):
        print("loading the model from %s" % generator_path)
        state_dict = torch.load(
            generator_path,
            map_location=str(self.device),
            weights_only=weights_only,
        )
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(
                state_dict, self.generator, key.split(".")
            )
        self.generator.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def predict(self, data):
        with torch.no_grad():
            predict = self.generator(data.to(self.device))

        return predict.detach().cpu()
