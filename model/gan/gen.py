import torch

def load_pix2pix_generator(model_file_path: str, gpu_ids: list = [], eval: bool = False):
    
    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=True,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=gpu_ids,
    )

    # get device name: CPU or GPU
    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    # if you are using PyTorch newer than 0.4, you can remove str() on self.device
    state_dict = torch.load(model_file_path, map_location=str(device))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    gen.load_state_dict(state_dict)

    # dropout and batchnorm has different behavioir during training and test.
    if eval:
        gen.eval()
    return gen