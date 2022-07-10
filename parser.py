
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--lr_crn_layer", type=float, default=5e-3, help="Learning rate for the CRN layer")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="netvlad", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool", "none"])
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--num_non_local', type=int, default=1, help="Num of non local blocks")
    parser.add_argument("--non_local", action='store_true', help="_")
    parser.add_argument('--channel_bottleneck', type=int, default=128, help="Channel bottleneck for Non-Local blocks")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument('--re_ranking', type=bool, default=False, choices=[True, False],
                        help="Re-Ranking / Post-Processing on n-top candidates prediction for each query")
    parser.add_argument('--re_ranking_cluster', type=int, default=3, 
                        help="number of clusters for Re-Ranking")
    
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--sharpness", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    parser.add_argument("--gan_augmentation", type=str, default="true", choices=["false", "true"] ,help="_")

    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default", help="Folder name of the current run (saved in ./logs/)")
   
    # GAN Params
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
    parser.set_defaults(load_size=parser.get_default('crop_size'))
    parser.add_argument('--dataroot', type=str, default='/tmp/pitts30k/images/train/queries/')
    parser.add_argument('--name', type=str, default='day2night_pretrained', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    # model parameters
    parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')


    args = parser.parse_args()
    
    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.aggregation == "crn" and args.resume == None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args

