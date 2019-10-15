#!/usr/bin/python

import argparse
import rosbag
import rospy
import os
import zipfile
import shutil
import sys
from os.path import basename


# Function borrowed from: https://stackoverflow.com/a/3041990
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def timestamp_str(ts):
    t = ts.secs + ts.nsecs / float(1e9)
    return '{:.12f}'.format(t)


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="ROS bag file to extract")
    parser.add_argument("--output_folder", default="extracted_data", help="Folder where to extract the data")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument('--no-zip', dest='no_zip', action='store_true')
    parser.set_defaults(no_zip=False)
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_folder))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    width, height = None, None
    event_sum = 0
    event_msg_sum = 0
    num_msgs_between_logs = 25
    output_name = os.path.basename(args.bag).split('.')[0]  # /path/to/mybag.bag -> mybag
    path_to_events_file = os.path.join(args.output_folder, '{}.txt'.format(output_name))

    with open(path_to_events_file, 'w') as events_file:

        with rosbag.Bag(args.bag, 'r') as bag:

            # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
            total_num_event_msgs = 0
            topics = bag.get_type_and_topic_info().topics
            for topic_name, topic_info in topics.iteritems():
                if topic_name == args.event_topic:
                    total_num_event_msgs = topic_info.message_count
                    print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))

            # Extract events to text file
            for topic, msg, t in bag.read_messages():
                if topic == args.event_topic:

                    if width is None:
                        width = msg.width
                        height = msg.height
                        print('Found sensor size: {} x {}'.format(width, height))
                        events_file.write("{} {}\n".format(width, height))

                    if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= total_num_event_msgs - 1:
                        print('Event messages: {} / {}'.format(event_msg_sum + 1, total_num_event_msgs))
                    event_msg_sum += 1

                    for e in msg.events:
                        events_file.write(timestamp_str(e.ts) + " ")
                        events_file.write(str(e.x) + " ")
                        events_file.write(str(e.y) + " ")
                        events_file.write(("1" if e.polarity else "0") + "\n")
                        event_sum += 1

        # statistics
        print('All events extracted!')
        print('Events:', event_sum)

    # Zip text file
    if not args.no_zip:
        print('Compressing text file...')
        path_to_events_zipfile = os.path.join(args.output_folder, '{}.zip'.format(output_name))
        with zipfile.ZipFile(path_to_events_zipfile, 'w') as zip_file:
            zip_file.write(path_to_events_file, basename(path_to_events_file), compress_type=zipfile.ZIP_DEFLATED)
        print('Finished!')

        # Remove events.txt
        if query_yes_no('Remove text file {}?'.format(path_to_events_file)):
            if os.path.exists(path_to_events_file):
                os.remove(path_to_events_file)
                print('Removed {}.'.format(path_to_events_file))

    print('Done extracting events!')
