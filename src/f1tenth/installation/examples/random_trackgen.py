# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Generates random tracks.
Adapted from https://gym.openai.com/envs/CarRacing-v0
Author: Hongrui Zheng

Note: additional requirements
    - shapely
    - opencv-python
"""
import os
import math
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shp


def main(args):
    outdir = os.path.dirname(os.path.abspath(__file__)) / args.outdir / args.name
    seed = args.seed

    np.random.seed(seed)

    outdir.mkdir(parents=True, exist_ok=True)

    checkpoints = None
    if os.path.isfile(args.checkpoints):
        import csv
        with open(args.checkpoints, 'r') as f:
            checkpoints = list(csv.reader(f))

    while True:
        try:
            print(f"[info] creating track {args.name}")
            track, track_int, track_ext = create_track(checkpoints)
            convert_track(track, track_int, track_ext, args.name, outdir)
            print(f"[info] saved track {args.name} in {outdir}/")
            break
        except Exception as _:  # noqa: F841
            print("[error] failed to create track. Retrying...")
            continue


def create_track(checkpoints=None):
    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 900 / SCALE
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31
    WIDTH = 7.0         # lanewidth

    start_alpha = 0.0

    # Create checkpoints
    '''
        alpha : random angle between 2pi * i ~ 2pi * (i+1)
        x, y : cos / sin position of alpha and radius which was randomly chosen
    '''
    if checkpoints is None:
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + np.random.uniform(
                0, 2 * math.pi * 1 / CHECKPOINTS
            )
            rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    else:
        # add radian angles
        for i, checkpoint in enumerate(checkpoints):
            cos, sin = float(checkpoint[0])/1.5, float(checkpoint[1])/1.5
            alpha = (np.angle(cos + 1j*sin) + 2*np.pi) % (2*np.pi)
            checkpoints[i] = [alpha, cos, sin]
        start_alpha = 2 * math.pi * (-0.5) / len(checkpoints)

    # Go from one checkpoint to another to create track
    x, y, beta = checkpoints[0][1], 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2 * math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    # Find closed loop
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return False
        pass_through_start = (
            track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        )
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break

    print("[info] track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))

    assert i1 != -1
    assert i2 != -1

    track = track[i1 : i2 - 1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)

    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # Converting to numpy array
    track_xy = [(x, y) for (a1, b1, x, y) in track]
    track_xy = np.asarray(track_xy)
    track_poly = shp.Polygon(track_xy)

    # Finding interior and exterior walls
    track_xy_offset_in = track_poly.buffer(WIDTH)
    track_xy_offset_out = track_poly.buffer(-WIDTH)
    track_xy_offset_in_np = np.array(track_xy_offset_in.exterior.xy).T
    track_xy_offset_out_np = np.array(track_xy_offset_out.exterior.xy).T

    return track_xy, track_xy_offset_in_np, track_xy_offset_out_np


def convert_track(track, track_int, track_ext, track_name, outdir):
    # converts track to image and saves the centerline as waypoints
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    ax.plot(track_int[:, 0], track_int[:, 1], color="black", linewidth=3)
    ax.plot(track_ext[:, 0], track_ext[:, 1], color="black", linewidth=3)
    plt.tight_layout()
    ax.set_aspect("equal")
    ax.set_xlim(-180, 300)
    ax.set_ylim(-300, 300)
    plt.axis("off")

    track_filepath = outdir / f"{track_name}_map.png"
    plt.savefig(track_filepath, dpi=80)
    plt.close()

    map_width, map_height = fig.canvas.get_width_height()
    print("[info] map image size: ", map_width, map_height)

    # Transform the track center line into pixel coordinates
    xy_pixels = ax.transData.transform(track)
    origin_x_pix = xy_pixels[0, 0]
    origin_y_pix = xy_pixels[0, 1]

    xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])

    map_origin_x = -origin_x_pix * 0.05
    map_origin_y = -origin_y_pix * 0.05

    # Convert image using cv2
    cv_img = cv2.imread(str(track_filepath), -1)
    cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Save image
    cv2.imwrite(str(track_filepath), cv_img_bw)
    cv2.imwrite(str(track_filepath.with_suffix(".pgm")), cv_img_bw)

    # create yaml file
    with open(track_filepath.with_suffix(".yaml"), "w") as yaml:
        yaml.write(f"image: {track_name}_map.pgm\n")
        yaml.write("resolution: 0.062500\n")
        yaml.write(f"origin: [{map_origin_x},{map_origin_y},0.000000]\n")
        yaml.write("negate: 0\n")
        yaml.write("occupied_thresh: 0.45\n")
        yaml.write("free_thresh: 0.196\n")
        yaml.write("width: 2.5")        # just for code completion

    # Saving centerline as a csv
    centerline_filepath = outdir / f"{track_name}_centerline.csv"
    with open(centerline_filepath, "w") as waypoints_csv:
        waypoints_csv.write("#x,y\n")
        for i, row in enumerate(xy_pixels):
            waypoints_csv.write(f"{0.05 * row[0]}, {0.05 * row[1]}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=123, help="The seed for the numpy."
    )
    parser.add_argument(
        "--outdir", type=pathlib.Path, default="../maps", help="Out directory."
    )
    parser.add_argument(
        "--name", type=str, default="map0", help="Name of map."
    )
    parser.add_argument(
        "--checkpoints", type=pathlib.Path, default='./', help="Checkpoints directory."
    )
    args = parser.parse_args()

    main(args)