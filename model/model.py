from base import BaseModel
import torch.nn as nn
import torch
from model.unet import UNet, UNetRecurrent
from os.path import join
from model.submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True


class E2VID(BaseE2VID):
    def __init__(self, config):
        super(E2VID, self).__init__(config)

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=1,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRecurrent, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states


class LegacyE2VID(BaseModel):
    """This is the E2VID model that was presented in the paper:
       "Events-to-Video: Bringing Modern Computer Vision to Event Cameras", CVPR'18 """

    def __init__(self, config):
        super(LegacyE2VID, self).__init__(config)

        self.num_input_channels = int(config['num_input_channels'])
        self.num_output_channels = int(config['num_output_channels'])
        try:
            self.use_upsample_conv = config['use_upsample_conv']
        except KeyError:
            self.use_upsample_conv = False

        if self.use_upsample_conv:
            print('Will use UpsampleConv (slow, but without checkerboard artefacts)')
        else:
            print('Will use TransposedConv (fast, but with checkerboard artefacts)')

        assert(self.num_input_channels > 0)

        self.conv1 = ConvLayer(self.num_input_channels, 64, 5, stride=2, padding=2, activation='relu', norm=None)
        self.conv2 = ConvLayer(64, 128, 5, stride=2, padding=2, activation='relu', norm=None)
        self.conv3 = ConvLayer(128, 256, 5, stride=2, padding=2, activation='relu', norm=None)
        self.conv4 = ConvLayer(256, 512, 5, stride=2, padding=2, activation='relu', norm=None)

        self.resblock1 = ResidualBlock(512, 512, norm='BN')
        self.resblock2 = ResidualBlock(512, 512, norm='BN')

        UpsampleLayer = UpsampleConvLayer if self.use_upsample_conv else TransposedConvLayer
        self.upsample1 = UpsampleLayer(512, 256, kernel_size=5, padding=2, norm=None, activation='relu')
        self.upsample2 = UpsampleLayer(self.num_output_channels + 256 + 256, 128,
                                       kernel_size=5, padding=2, norm=None, activation='relu')
        self.upsample3 = UpsampleLayer(self.num_output_channels + 128 + 128, 64,
                                       kernel_size=5, padding=2, norm=None, activation='relu')
        self.upsample4 = UpsampleLayer(self.num_output_channels + 64 + 64, 32,
                                       kernel_size=5, padding=2, norm=None, activation='relu')

        self.pred1 = ConvLayer(256, self.num_output_channels, kernel_size=1, norm=None, activation=None)
        self.pred2 = ConvLayer(128, self.num_output_channels, kernel_size=1, norm=None, activation=None)
        self.pred3 = ConvLayer(64, self.num_output_channels, kernel_size=1, norm=None, activation=None)

        self.pred4 = ConvLayer(32 + self.num_input_channels, self.num_output_channels,
                               kernel_size=1, norm=None, activation=None)

    def forward(self, x, last_states=None):
        """
        :param x: N x num_input_channels x H x W
        :return: N x 1 x H x W
        """

        # downsample
        x1 = self.conv1(x)   # N x 64 x H x W
        x2 = self.conv2(x1)  # N x 128 x H/2 x W/2
        x3 = self.conv3(x2)  # N x 256 x H/4 x W/4
        x4 = self.conv4(x3)  # N x 512 x H/8 x W/8

        # residual blocks
        x5 = self.resblock1(x4)  # N x 512 x H/16 x W/16
        x6 = self.resblock2(x5)  # N x 512 x H/16 x W/16

        # upsampling
        x7 = self.upsample1(x6)    # N x 256 x H/8 x W/8
        img1 = self.pred1(x7)             # N x num_output_channels x H/8 x W/8

        x8 = torch.cat((x3, x7, img1), 1)  # N x (num_output_channels + 256 + 256) x H/8 x W/8

        x9 = self.upsample2(x8)
        img2 = self.pred2(x9)

        x10 = torch.cat((x2, x9, img2), 1)

        x11 = self.upsample3(x10)
        img3 = self.pred3(x11)

        x12 = torch.cat((x1, x11, img3), 1)
        x13 = self.upsample4(x12)

        x14 = torch.cat((x, x13), 1)
        img4 = self.pred4(x14)

        return img4, None
