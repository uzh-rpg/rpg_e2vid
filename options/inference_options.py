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

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image. \
                              Useful to hide boundary effects when safety_margin = 0.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    parser.add_argument('--safety_margin', default=5, type=int,
                        help="Safety margin. The input event tensors are padded with zeros to avoid boundary effects. \
                              Range: [0, 5]. A small value reduces computation time but may introduce boundary effects")

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

    """ Automatic rescaling of the image intensities to increase the image contrast """
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true')
    parser.set_defaults(auto_hdr=False)

    parser.add_argument('--auto_hdr_min_percentile', default=0.5, type=float,
                        help="Percentile to use for robust min in auto_hdr mode. Should lie in the range [0, 30]. Higher means more contrast but more artefacts.")
    parser.add_argument('--auto_hdr_max_percentile', default=99.5, type=float,
                        help="Percentile to use for robust max in auto_hdr mode. Should lie in the range [70, 100]. Lower means more contrast but more artefacts.")
    parser.add_argument('--auto_hdr_border', default=5, type=int,
                        help="Width of the outer border to exclude when computing the bounds in the auto HDR mode (to avoid boundary effects).")
    parser.add_argument('--auto_hdr_moving_average_size', default=5, type=int,
                        help="Size of the moving average window used to filter the robust min/max used for auto HDR (range: [1, 100]). \
                              Small value: less lag (but potential flicker), large value: a bit of lag but less flicker).")

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
