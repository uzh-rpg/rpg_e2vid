def set_inference_options(parser):

    parser.add_argument('-o', '--output_folder', default=None, type=str)  # if None, will not write the images to disk
    parser.add_argument('--dataset_name', default='reconstruction', type=str)

    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)

    """ Display """
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)

    parser.add_argument('--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=False)

    parser.add_argument('--event_display_mode', default='red-blue', type=str,
                        help="Event display mode ('red-blue' or 'grayscale')")

    parser.add_argument('--num_bins_to_show', default=-1, type=int,
                        help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins).")

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    """ Post-processing / filtering """

    # (optional) path to a text file containing the locations of hot pixels to ignore
    parser.add_argument('--hot_pixels_file', default=None, type=str)

    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.set_defaults(flip=False)

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")

    """ Perform color reconstruction? (only use this flag with the DAVIS346color) """
    parser.add_argument('--color', dest='color', action='store_true')
    parser.set_defaults(color=False)

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')
    parser.set_defaults(no_normalize=False)

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')
    parser.set_defaults(no_recurrent=False)
