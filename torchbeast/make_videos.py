"""
Purpose:
    Load video + attention maps from trained model. Create video.
"""

import numpy as np
import os
import skvideo.io

def main(directory):
    vid_filename = os.path.join(directory, "vid_array.npy")
    att_filename = os.path.join(directory, "attention_array.npy")
    real_vid_filename = os.path.join(directory, "videp.mp4")

    with open(vid_filename, "rb") as f:
        vid_frames = np.load(f)

    with open(att_filename, "rb") as f:
        att_frames = np.load(f)

    augmented_frames = make_video(vid_frames, att_frames)
    augmented_frames *= 255
    augmented_frames = augmented_frames.astype(np.uint8)

    skvideo.io.vwrite(real_vid_filename, augmented_frames)



def make_video(video_frames, attention_frames):
    """
    
    """

    print(f"Vid shape: {video_frames.shape}")
    print(f"Att shape: {attention_frames.shape}")

    print(f"What data type are we?: {video_frames.dtype} and {attention_frames.shape}")
    print(f"what's the min and max in video_frames? {video_frames.min()} and {video_frames.max()}")
    print(f"what's the min and max in attention? {attention_frames.min()} and {attention_frames.max()}")

    assert video_frames.min() >= 0.0, "The way we have it assumes (0-1) bounded float videos"
    assert video_frames.max() <= 1.0, "The way we have it assumes (0-1) bounded float videos"

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


def make_frame_from_one_attention_head(video_frame, attention_head):
    """
    Either straight highlighting:
        video_frame * attention_head
    or with dark frame as background
        0.1 * video_frame + 0.9 * video_frame * attention_head
    """

    augmented_frame = np.zeros_like(video_frame)

    for c in range(3):
        channel = video_frame[:,:,c]
        print(attention_head)
        augmented_frame[:,:,c] = video_frame[:,:,c] * attention_head

    return augmented_frame


def _write_video(directory):
    num_frames = 100
    height = 210
    width = 160
    num_attention = 8


    video_frames = np.zeros(dtype=np.float32, shape=(num_frames, height, width, 3))
    video_frames[:,:,:,0] = 1.0
    attention_frames = np.random.uniform(low=0.0, high=1.0, size=(num_frames, height, width, 8))

    with open(os.path.join(directory, "vid_array.npy"), "wb") as f:
        np.save(f, video_frames)

    with open(os.path.join(directory, "attention_array.npy"), "wb") as f:
        np.save(f, attention_frames)


if __name__ == "__main__":
    main(directory="/home/sam/Code/ML/INTERPRETABLE_RL/torchbeast_charlie/test_stuff")
    # _write_video(directory="/home/sam/Code/ML/INTERPRETABLE_RL/torchbeast_charlie/test_stuff")
