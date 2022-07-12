from email.policy import default
from model.gan.options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, pr):
        print('TestOptions start')
        pr = BaseOptions.initialize(self, pr)  # define shared options
        pr.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        pr.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        pr.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        pr.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        pr.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
     #   parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        pr.set_defaults(load_size=pr.get_default('crop_size'))
        self.isTrain = False
        print('TestOptions end')
        return pr