import torch
import cv2
import numpy as np
from model.model import *
from utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, ImageWriter, UnsharpMaskFilter
from utils.inference_utils import upsample_color_image, merge_channels_into_color_image  # for color reconstruction
from utils.util import robust_min, robust_max
from utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque
import torch.nn.functional as F


class ImageReconstructor:
    def __init__(self, model, height, width, num_bins, options):

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.height = height
        self.width = width
        self.num_bins = num_bins

        self.initialize(self.height, self.width, options)

    def initialize(self, height, width, options):
        print('== Image reconstruction == ')
        print('Image size: {}x{}'.format(self.height, self.width))

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.perform_color_reconstruction = options.color  # whether to perform color reconstruction (only use this with the DAVIS346color)
        if self.perform_color_reconstruction:
            if options.auto_hdr:
                print('!!Warning: disabling auto HDR for color reconstruction!!')
            options.auto_hdr = False  # disable auto_hdr for color reconstruction (otherwise, each channel will be normalized independently)

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)

        self.last_states_for_each_channel = {'grayscale': None}

        if self.perform_color_reconstruction:
            self.crop_halfres = CropParameters(int(width / 2), int(height / 2),
                                               self.model.num_encoders)
            for channel in ['R', 'G', 'B', 'W']:
                self.last_states_for_each_channel[channel] = None

        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.image_writer = ImageWriter(options)
        self.image_display = ImageDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):
        with torch.no_grad():

            with CudaTimer('Reconstruction'):

                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor.unsqueeze(dim=0)
                    events = events.to(self.device)

                events = self.event_preprocessor(events)

                # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                events_for_each_channel = {'grayscale': self.crop.pad(events)}
                reconstructions_for_each_channel = {}
                if self.perform_color_reconstruction:
                    events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
                    events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
                    events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
                    events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])

                # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
                for channel in events_for_each_channel.keys():
                    with CudaTimer('Inference'):
                        new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                                 self.last_states_for_each_channel[channel])

                    if self.no_recurrent:
                        self.last_states_for_each_channel[channel] = None
                    else:
                        self.last_states_for_each_channel[channel] = states

                    # Output reconstructed image
                    crop = self.crop if channel == 'grayscale' else self.crop_halfres

                    # Unsharp mask (on GPU)
                    new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                    # Intensity rescaler (on GPU)
                    new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                    with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                        crop.ix0:crop.ix1].cpu().numpy()

                if self.perform_color_reconstruction:
                    out = merge_channels_into_color_image(reconstructions_for_each_channel)
                else:
                    out = reconstructions_for_each_channel['grayscale']

            # Post-processing, e.g bilateral filter (on CPU)
            out = self.image_filter(out)

            self.image_writer(out, event_tensor_id, stamp, events=events)
            self.image_display(out, events)
