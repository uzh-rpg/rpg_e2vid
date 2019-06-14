import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from os.path import join
import rospy
import argparse
import shutil
import os
import glob

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default='dynamic_6dof', type=lambda s: [str(item) for item in s.split(',')],
                        help="Delimited list of datasets")
    parser.add_argument("--image_folder", required=True,
                        type=str, help="Path to the base folder containing the image reconstructions")
    parser.add_argument("--output_folder", default='.',
                        type=str, help="Path to the output folder")
    parser.add_argument("--image_topic", required=True, type=str,
                        help="Name of the topic which will contain the reconstructed images")
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help="Whether to overwrite existing rosbags (default: false)")
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    print('Datasets to process: {}'.format(args.datasets))

    for dataset in args.datasets:
        reconstructed_images_folder = join(
            args.image_folder, dataset)

        bridge = CvBridge()
        continue_processing = True

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        output_bag_filename = join(
            args.output_folder, '{}.bag'.format(dataset))

        if continue_processing:
            # Write the images to a rosbag
            stamps = np.loadtxt(
                join(reconstructed_images_folder, 'timestamps.txt'))
            if len(stamps.shape) == 2:
                stamps = stamps[:, 1]

            # list all images in the folder
            images = [f for f in glob.glob(join(reconstructed_images_folder, "*.png"))]
            images = sorted(images)
            print('Found {} images'.format(len(images)))

            with rosbag.Bag(output_bag_filename, 'w') as outbag:

                for i, image_path in enumerate(images):

                    stamp = stamps[i]
                    img = cv2.imread(join(reconstructed_images_folder, image_path), 0)

                    try:
                        img_msg = bridge.cv2_to_imgmsg(img, encoding='mono8')
                        stamp_ros = rospy.Time(stamp)
                        print(img.shape, stamp_ros)
                        img_msg.header.stamp = stamp_ros
                        img_msg.header.seq = i
                        outbag.write(args.image_topic, img_msg,
                                     img_msg.header.stamp)

                    except CvBridgeError, e:
                        print e
