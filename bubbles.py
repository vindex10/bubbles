import os
import numpy as np
from numpy import random
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

DIM = 2
MAX_NUM = 1000
BUBBLING_RATE = 25
GROWTH_SPEED = 0.001
ROOM_SIZE = 1
dt = 0.01
steps = 0
os.makedirs("data", exist_ok=True)


def main():
    num_bubbles = [0]
    center = np.zeros((DIM, MAX_NUM))
    radius = np.zeros(MAX_NUM)
    growth_speed = np.zeros(MAX_NUM)

    while np.any(growth_speed) or num_bubbles[0] < MAX_NUM:
        step(num_bubbles, center, radius, growth_speed)


def step(num_bubbles, center, radius, growth_speed):
    global steps
    growth(radius, growth_speed)
    freezing(num_bubbles, center, radius, growth_speed)
    new_bubbles(num_bubbles, center, radius, growth_speed)
    steps += 1


def growth(radius, growth_speed):
    mask = growth_speed > 0
    radius[mask] += growth_speed[mask]*dt


def freezing(num_bubbles, center, radius, growth_speed):
    print(num_bubbles, np.count_nonzero(growth_speed), np.sum(np.pi*radius**2))
    if steps % 25 == 0:
        for i, j in zip(center.T[:num_bubbles[0]], radius.T[:num_bubbles[0]]):
            plt.gca().add_patch(plt.Circle(i, j, facecolor="r", edgecolor="b"))
        plt.savefig(f"data/{steps//25}.png")
        plt.close()
        np.savetxt("data/center.txt", center)
        np.savetxt("data/radius.txt", radius)

    mask = slice(None, num_bubbles[0])
    valid_mask = check_valid_centers(center[:, mask], center[:, mask], radius[mask], radius[mask], ignore_diag=True)
    growth_speed[mask][~valid_mask] = 0


def new_bubbles(num_bubbles, center, radius, growth_speed):
    extra_num = sp.stats.poisson(mu=BUBBLING_RATE*dt).rvs(1)[0]
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


if __name__ == "__main__":
    main()
