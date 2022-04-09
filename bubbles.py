import os
import numpy as np
from numpy import random
import h5py
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

DIM = 2
INIT_MAX_NUM = 1000
BUBBLING_RATE = 25
GROWTH_SPEED = 0.001
ROOM_SIZE = 1
DT = 0.01
_STEPS = 0
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TerminateBubbles(Exception):
    pass


def main():
    num_bubbles = [0]
    center = np.zeros((DIM, INIT_MAX_NUM))
    radius = np.zeros(INIT_MAX_NUM)
    growth_speed = np.zeros(INIT_MAX_NUM)
    emerge_times = np.zeros(INIT_MAX_NUM)

    with open(f"{OUTPUT_DIR}/stats.dat", "w", encoding="utf-8") as stats_f:
        while True:
            try:
                step(num_bubbles, center, radius, growth_speed, emerge_times, stats_f)
            except TerminateBubbles:
                break

    with h5py.File(f"{OUTPUT_DIR}/final_state.h5", "w") as fout:
        final_dump(fout, num_bubbles, center, radius, growth_speed, emerge_times)


def step(num_bubbles, center, radius, growth_speed, emerge_times, stats_f):
    global _STEPS  # pylint: disable=global-statemenet
    growth(radius, growth_speed)
    freezing(num_bubbles, center, radius, growth_speed)
    do_logs(stats_f, num_bubbles, center, radius, growth_speed)
    new_bubbles(num_bubbles, center, radius, growth_speed, emerge_times)
    _STEPS += 1


def growth(radius, growth_speed):
    mask = growth_speed > 0
    radius[mask] += growth_speed[mask]*DT


def freezing(num_bubbles, center, radius, growth_speed):
    mask = slice(None, num_bubbles[0])
    valid_mask = check_valid_centers(center[:, mask], center[:, mask], radius[mask], radius[mask], ignore_diag=True)
    growth_speed[mask][~valid_mask] = 0


def new_bubbles(num_bubbles, center, radius, growth_speed, emerge_times):
    extra_num = sp.stats.poisson(mu=BUBBLING_RATE*DT).rvs(1)[0]
    if extra_num + num_bubbles[0] > center.shape[1]:
        raise TerminateBubbles()
    if extra_num == 0:
        return
    new_centers = []
    mask = slice(None, num_bubbles[0])
    while len(new_centers) < extra_num:
        new_center = sample_one_center()
        valid_centers = check_valid_centers(center[:, mask], new_center, radius[mask], np.array([0]))
        if not valid_centers[0]:
            continue
        new_centers.append(new_center)
    new_centers = np.hstack(new_centers)
    center[:, num_bubbles[0]:(num_bubbles[0]+extra_num)] = new_centers
    growth_speed[num_bubbles[0]:(num_bubbles[0]+extra_num)] = GROWTH_SPEED
    emerge_times[num_bubbles[0]:(num_bubbles[0]+extra_num)] = _STEPS
    num_bubbles[0] += extra_num


def sample_one_center():
    new_center = np.random.random((DIM, 1))*ROOM_SIZE
    return new_center


def check_valid_centers(centers_a, centers_b, radius_a, radius_b, ignore_diag=False):
    # if centers_a.shape[1] > 300:
        # import ipdb; ipdb.set_trace()
    if centers_a.shape[1] == 0:
        return np.ones(centers_b.shape[1]).astype(bool)
    dist = np.square(centers_a.T[:, np.newaxis, :] - centers_b.T).sum(axis=-1)
    radii = (radius_a[:, np.newaxis] + radius_b)**2
    check = dist > radii
    if ignore_diag:
        check[np.eye(check.shape[0], check.shape[1]).astype(bool)] = True
    vacant = np.all(check, axis=0)
    return vacant


def do_logs(stats_f, num_bubbles, center, radius, growth_speed):
    stats_data = {"num_bubbles": num_bubbles[0],
                  "active_bubbles": np.count_nonzero(growth_speed),
                  "total_area": np.sum(np.pi*radius**2)}
    line = "\t".join(map(str, stats_data.values()))
    print(line)
    stats_f.write(line + "\n")
    if _STEPS % 25 == 0:
        plt.gca().set_aspect(1)
        for i, j in zip(center.T[:num_bubbles[0]], radius.T[:num_bubbles[0]]):
            plt.gca().add_patch(plt.Circle(i, j, facecolor="r", edgecolor="b"))
        plt.savefig(f"{OUTPUT_DIR}/{_STEPS//25}.png")
        plt.close()


def final_dump(final_f, num_bubbles, center, radius, growth_speed, emerge_times):
    final_f["center"] = center
    final_f["radius"] = radius
    final_f["growth_speed"] = growth_speed
    final_f["emerge_times"] = emerge_times
    final_f.attrs["DIM"] = DIM
    final_f.attrs["INIT_MAX_NUM"] = INIT_MAX_NUM
    final_f.attrs["BUBBLING_RATE"] = BUBBLING_RATE
    final_f.attrs["GROWTH_SPEED"] = GROWTH_SPEED
    final_f.attrs["ROOM_SIZE"] = ROOM_SIZE
    final_f.attrs["DT"] = DT
    final_f.attrs["num_bubbles"] = num_bubbles[0]
    final_f.attrs["_STEPS"] = _STEPS


if __name__ == "__main__":
    main()
