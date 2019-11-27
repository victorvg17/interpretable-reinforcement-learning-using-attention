"""
Purpose:
    Load video + attention maps from trained model. Create video.
"""

import numpy as np
import os
import skvideo.io

import argparse
import logging

# We want to load the same way we save...
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")



def main(directory):
    """
    Loads video and attention arrays, combines them together, and then writes to a video file
    in the same directory.

    Arguments:
        directory:
            The directory created by torchbeast to store models and logs

    TODO: The video could be improved. Also, it would be nice to choose the head, or make a video
          with more than one head. Also, maybe naming the video to keep track of different videos
          from the same experiment.
    """
    vid_filename = os.path.join(directory, "vid_array.npy")
    att_filename = os.path.join(directory, "attention_array.npy")
    real_vid_filename = os.path.join(directory, "video.mp4")

    with open(vid_filename, "rb") as f:
        vid_frames = np.load(f)

    with open(att_filename, "rb") as f:
        att_frames = np.load(f)

    augmented_frames = make_video(vid_frames, att_frames)
    # augmented_frames *= 255
    augmented_frames = augmented_frames.astype(np.uint8)

    skvideo.io.vwrite(real_vid_filename, augmented_frames)



def make_video(video_frames, attention_frames):
    """
    http://www.scikit-video.org/stable/io.html#writing
    """

    print(f"Vid shape: {video_frames.shape}")
    print(f"Att shape: {attention_frames.shape}")

    print(f"What data type are we?: {video_frames.dtype} and {attention_frames.shape}")
    print(f"what's the min and max in video_frames? {video_frames.min()} and {video_frames.max()}")
    print(f"what's the min and max in attention? {attention_frames.min()} and {attention_frames.max()}")

    # assert video_frames.min() >= 0.0, "The way we have it assumes (0-1) bounded float videos"
    # assert video_frames.max() <= 1.0, "The way we have it assumes (0-1) bounded float videos"

    print(video_frames.shape)
    print(attention_frames.shape)

    assert video_frames.shape[-1] == 3 # 3 channels!

    assert video_frames.shape[0:2] == attention_frames.shape[0:2]
    assert video_frames.shape[1:3] == (210, 160)

    augmented_frames = np.zeros_like(video_frames)

    num_frames = video_frames.shape[0]
    for i in range(num_frames):
        frame = video_frames[i]
        head = attention_frames[i]
        new_frame = make_frame(frame, head)
        augmented_frames[i,:,:,:] = new_frame

    return augmented_frames


def make_frame(video_frame, attention_frame):
    first_head = attention_frame[:,:,0]
    return make_frame_from_one_attention_head(video_frame, first_head)


def scale_attention_head(attention_head):
    """
    Attention head sums to 1, meaning every entry is very small.
    We don't want to make a black image, so we scale it so its min
    is 0 and and its max is 1.
    """
    min_val = attention_head.min()
    max_val = attention_head.max()
    attention_head = (attention_head - min_val) / (max_val - min_val)
    return attention_head

def make_frame_from_one_attention_head(video_frame, attention_head):
    """
    Turns a frame and an attention head to a highlighted attention head.
    Right now it works by just elem-multiplying with a scaled attention
    head.
    """

    attention_head = scale_attention_head(attention_head)

    augmented_frame = np.zeros_like(video_frame)

    for c in range(3):
        channel = video_frame[:,:,c]
        augmented_frame[:,:,c] = video_frame[:,:,c] * attention_head

    return augmented_frame

if __name__ == "__main__":
    # Creates the directory the same way that torchbeast does.
    flags = parser.parse_args()
    directory = os.path.expandvars(
            os.path.expanduser("%s/%s" % (flags.savedir, flags.xpid)))

    main(directory=directory)
