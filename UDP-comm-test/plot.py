# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum, auto


class Condition(Enum):
    """
    Enum class for the condition of the data
    """

    C1 = 1  # Working alone
    C2 = auto()  # Resting alone
    C3 = auto()  # Walking alone
    C4 = auto()  # Working together
    C5 = auto()  # Resting together
    C6 = auto()  # Walking together


class Participant(Enum):
    """
    Enum class for the participants
    """

    P1 = 1
    # P2 = auto()
    # P3 = auto()
    # P4 = auto()
    # P5 = auto()
    # P6 = auto()


# %%
participants = map(lambda p: p.name, Participant)  # 6 patients
conditions = map(lambda c: c.name, Condition)  # 6 conditions := labels

# 6 patients x 6 conditions index
# P1_C1, P1_C2, P1_C3, P1_C4, P1_C5, P1_C6 ...
index = pd.MultiIndex.from_product([participants, conditions]).map("_".join)

# labels are the conditions
# [1, 2, 3, 4, 5, 6] * 6
labels = pd.Series(
    index=index, data=len(Participant) * list(map(lambda c: c.value, Condition))
)

# %%
dir = os.path.dirname(os.path.realpath(__file__))
csv_files = []
for root, dirs, files in os.walk(dir + "/data"):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))
csv_files.sort()

# %%
def auc_activity(data, left=0, right=np.inf):
    """
    Calculates the area under the curve of the data
    """
    data = data[data.timestamp.between(left, right, inclusive="both")]

    return np.trapz(data.activity, x=data.timestamp)


def avg_activity(data, left=0, right=np.inf):
    """
    Calculates the average of the activity
    """
    data = data[data.timestamp.between(left, right, inclusive="both")]
    interval = data.timestamp.max() - data.timestamp.min()
    return auc_activity(data, left, right) / interval


def read_activity(file):
    """
    Reads the activity data from the file
    """
    data = pd.read_csv(file, header=None, names=["timestamp", "activity"])
    return data


# %%
desktop_csvfiles = [f for f in csv_files if "desktop" in f]
mobile_csvfiles = [f for f in csv_files if "phone" in f]
laptop_csvfiles = [f for f in csv_files if "laptop" in f]
activities_json = {}
for idx in index:
    desktop_csvfile, *_ = [f for f in desktop_csvfiles if idx in f]
    mobile_csvfile, *_ = [f for f in mobile_csvfiles if idx in f]
    laptop_csvfile, *_ = [f for f in laptop_csvfiles if idx in f]
    desktop_data = read_activity(desktop_csvfile)
    mobile_data = read_activity(mobile_csvfile)
    laptop_data = read_activity(laptop_csvfile)
    activities_json[idx] = {
        "desktop": desktop_data,
        "desktop_avg": avg_activity(desktop_data),
        "desktop_auc": auc_activity(desktop_data),
        "mobile": mobile_data,
        "mobile_avg": avg_activity(mobile_data),
        "mobile_auc": auc_activity(mobile_data),
        "laptop": laptop_data,
        "laptop_avg": avg_activity(laptop_data),
        "laptop_auc": auc_activity(laptop_data),
    }

activities = pd.DataFrame(activities_json).T.convert_dtypes()
activities["label"] = labels


def read_face(file):
    """
    Reads the face data from the file
    """
    data = pd.read_csv(file, header=0, names=["timestamp", "face"])
    return data


def read_imu(file):
    """
    Reads the imu data from the file
    """
    data = pd.read_csv(
        file,
        header=0,
        names=[
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
        ],
    )
    return data


def read_posture(file):
    """
    Reads the posture data from the file
    """
    data = pd.read_csv(file, header=0, names=["timestamp", "posture"], usecols=range(2))
    return data


hololens_face_csvfiles = [f for f in csv_files if "Faces" in f]
hololens_imu_csvfiles = [f for f in csv_files if "IMU" in f]
hololens_posture_csvfiles = [f for f in csv_files if "Posture" in f]


# read hololens data
hololens_face_json = {}
hololens_imu_json = {}
hololens_posture_json = {}
for idx in index:
    hololens_face_csvfile, *_ = [f for f in hololens_face_csvfiles if idx in f]
    hololens_imu_csvfile, *_ = [f for f in hololens_imu_csvfiles if idx in f]
    hololens_posture_csvfile, *_ = [f for f in hololens_posture_csvfiles if idx in f]
    hololens_face_data = read_face(hololens_face_csvfile)
    hololens_imu_data = read_imu(hololens_imu_csvfile)
    hololens_posture_data = read_posture(hololens_posture_csvfile)
    hololens_face_json[idx] = {
        "timestamp": hololens_face_data.timestamp,
        "face": hololens_face_data.face,
    }
    hololens_imu_json[idx] = {
        "timestamp": hololens_imu_data.timestamp,
        "accel_x": hololens_imu_data.accel_x,
        "accel_y": hololens_imu_data.accel_y,
        "accel_z": hololens_imu_data.accel_z,
        "gyro_x": hololens_imu_data.gyro_x,
        "gyro_y": hololens_imu_data.gyro_y,
        "gyro_z": hololens_imu_data.gyro_z,
        "mag_x": hololens_imu_data.mag_x,
        "mag_y": hololens_imu_data.mag_y,
        "mag_z": hololens_imu_data.mag_z,
    }
    hololens_posture_json[idx] = {
        "timestamp": hololens_posture_data.timestamp,
        "posture": hololens_posture_data.posture,
    }

hololens_face = pd.DataFrame(hololens_face_json).T.convert_dtypes()
hololens_imu = pd.DataFrame(hololens_imu_json).T.convert_dtypes()
hololens_posture = pd.DataFrame(hololens_posture_json).T.convert_dtypes()


def read_empatica(file):
    """
    Reads the empatica data from the file
    """
    data = pd.read_csv(
        file,
        header=0,
        names=[
            "eda",
            "bvp",
            "temp",
            "acc_x",
            "acc_y",
            "acc_z",
            "acc",
            "ibi_sum",
            "beats",
        ],
        usecols=range(3, 12),
    )
    return data


empatica_csvfiles = [f for f in csv_files if "empatica" in f]

empatica_json = {}
for idx in index:
    empatica_csvfile, *_ = [f for f in empatica_csvfiles if idx in f]
    empatica_data = read_empatica(empatica_csvfile)
    empatica_json[idx] = {
        "timestamp": pd.Series(range(len(empatica_data))),
        "eda": empatica_data.eda,
        "bvp": empatica_data.bvp,
        "temp": empatica_data.temp,
        "acc_x": empatica_data.acc_x,
        "acc_y": empatica_data.acc_y,
        "acc_z": empatica_data.acc_z,
        "acc": empatica_data.acc,
        "ibi_sum": empatica_data.ibi_sum,
        "beats": empatica_data.beats,
    }

empatica = pd.DataFrame(empatica_json).T.convert_dtypes()

# %%
from matplotlib.animation import FuncAnimation

# plot empatica data for a single subject as an animation gif
# this is just to show how the data looks like

for idx in index:
    fig, ax = plt.subplots(5, 1, figsize=(15, 10), sharex=True)

    def animate(i):
        ax[0].set_title(f"EDA {idx}")
        ax[0].set_ylabel("EDA")
        ax[0].set_ylim(
            -0.1 + empatica.loc[idx, "eda"].min(), 0.1 + empatica.loc[idx, "eda"].max()
        )
        ax[0].set_xlim(0, empatica.loc[idx, "timestamp"].max())
        ax[0].plot(
            empatica.loc[idx, "timestamp"][:i],
            empatica.loc[idx, "eda"][:i],
            color="purple",
        )

        ax[1].set_title(f"BVP {idx}")
        ax[1].set_ylabel("BVP")
        ax[1].set_ylim(
            -0.1 + empatica.loc[idx, "bvp"].min(), 0.1 + empatica.loc[idx, "bvp"].max()
        )
        ax[1].set_xlim(0, empatica.loc[idx, "timestamp"].max())
        ax[1].plot(
            empatica.loc[idx, "timestamp"][:i],
            empatica.loc[idx, "bvp"][:i],
            color="green",
        )

        ax[2].set_title(f"Temperature {idx}")
        ax[2].set_ylabel("Temperature")
        ax[2].set_ylim(
            -0.1 + empatica.loc[idx, "temp"].min(),
            0.1 + empatica.loc[idx, "temp"].max(),
        )
        ax[2].set_xlim(0, empatica.loc[idx, "timestamp"].max())
        ax[2].plot(
            empatica.loc[idx, "timestamp"][:i],
            empatica.loc[idx, "temp"][:i],
            color="blue",
        )

        ax[3].set_title(f"Sum of IBI {idx}")
        ax[3].set_ylabel("Sum of IBI")
        ax[3].set_ylim(
            -0.1 + empatica.loc[idx, "ibi_sum"].min(),
            0.1 + empatica.loc[idx, "ibi_sum"].max(),
        )
        ax[3].set_xlim(0, empatica.loc[idx, "timestamp"].max())
        ax[3].plot(
            empatica.loc[idx, "timestamp"][:i],
            empatica.loc[idx, "ibi_sum"][:i],
            color="orange",
        )

        ax[4].set_title(f"Beats {idx}")
        ax[4].set_ylabel("Beats")
        ax[4].set_ylim(
            -0.1 + empatica.loc[idx, "beats"].min(),
            0.1 + empatica.loc[idx, "beats"].max(),
        )
        ax[4].set_xlim(0, empatica.loc[idx, "timestamp"].max())
        ax[4].plot(
            empatica.loc[idx, "timestamp"][:i],
            empatica.loc[idx, "beats"][:i],
            color="red",
        )
        ax[4].set_xlabel("Time (s)")
        plt.tight_layout()
        print(".", end="")

    # 30 seconds of data
    anim = FuncAnimation(fig, animate, frames=range(0, 3000, 10))
    anim.save(f"gifs/empatica_{idx}.gif", writer="imagemagick", fps=30)
    print(f"Saved empatica_{idx}.gif")


# %%


def animate(i, x, y, ax, ylabel, title):
    ax.clear()
    ax.plot(x[:i], y[:i])
    ax.set_ylim(-0.1, np.ceil(y.max()) + 0.1)
    ax.set_xlim(0, 30 + 0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("{}".format(ylabel))
    ax.set_title(f"{title} (t={x[i]:.2f}s)")
    print(".", end="")
    return ax


for idx in index:
    # hololens face
    try:
        fig, ax = plt.subplots()
        x = hololens_face.timestamp[idx]
        y = hololens_face.face[idx]
        anim = FuncAnimation(
            fig,
            animate,
            frames=range(0, 30 * 30),
            fargs=(x, y, ax, "Face", f"Hololens Face {idx}"),
        )
        anim.save(f"gifs/hololens_face_{idx}.gif", writer="imagemagick", fps=10)
        print(f"Saved hololens face {idx} as gif")
    except:
        print(f"Hololens Face data for {idx} not available")

    # hololens posture
    try:
        fig, ax = plt.subplots()
        x = hololens_posture.timestamp[idx]
        y = hololens_posture.posture[idx]
        anim = FuncAnimation(
            fig,
            animate,
            frames=range(0, 30 * 30),
            fargs=(x, y, ax, "Posture", f"Hololens Posture {idx}"),
        )
        anim.save(f"gifs/hololens_posture_{idx}.gif", writer="imagemagick", fps=10)
        print(f"Saved hololens posture {idx} as gif")
    except:
        print(f"Hololens Posture data for {idx} not available")
# %%
# hololens imu data animation in 3d space
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def animate(i, t, x, y, z, ax, title):
    ax.clear()
    ax.plot3D(x[:i], y[:i], z[:i])
    ax.set_xlim(np.floor(x.min()) - 0.1, np.ceil(x.max()) + 0.1)
    ax.set_ylim(np.floor(y.min()) - 0.1, np.ceil(y.max()) + 0.1)
    ax.set_zlim(np.floor(z.min()) - 0.1, np.ceil(z.max()) + 0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title} (t={t[i]:.2f}s)")
    print(".", end="")
    return ax


for idx in index:
    # hololens imu
    # try:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    t = hololens_imu.timestamp[idx]
    x = hololens_imu.accel_x[idx]
    y = hololens_imu.accel_y[idx]
    z = hololens_imu.accel_z[idx]
    anim = FuncAnimation(
        fig,
        animate,
        frames=range(0, 30 * 30),
        fargs=(t, x, y, z, ax, f"Hololens IMU {idx}"),
    )
    anim.save(f"gifs/hololens_imu_{idx}.gif", writer="imagemagick", fps=10)
    print(f"Saved hololens imu {idx} as gif")
# except:
#     print(f"Hololens IMU data for {idx} not available")


# %%
# plot time series data animation on desktop
from matplotlib.animation import FuncAnimation


for idx in index:
    try:
        fig, ax = plt.subplots()
        x = activities.loc[idx, "desktop"].timestamp
        y = activities.loc[idx, "desktop"].activity
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(x) + 1,
            fargs=(x, y, ax, "Activity", f"Desktop Activity {idx}"),
        )
        anim.save(f"gifs/desktop_activity_{idx}.gif", writer="imagemagick", fps=10)
    except:
        print(f"Desktop data for {idx} not available")

    try:
        fig, ax = plt.subplots()
        x = activities.loc[idx, "mobile"].timestamp
        y = activities.loc[idx, "mobile"].activity
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(x) + 1,
            fargs=(x, y, ax, "Activity", f"Mobile Activity {idx}"),
        )
        anim.save(f"gifs/mobile_activity_{idx}.gif", writer="imagemagick", fps=10)
    except:
        print(f"Mobile data for {idx} not available")

    try:
        fig, ax = plt.subplots()
        x = activities.loc[idx, "laptop"].timestamp
        y = activities.loc[idx, "laptop"].activity
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(x) + 1,
            fargs=(x, y, ax, "Activity", f"Laptop Activity {idx}"),
        )
        anim.save(f"gifs/laptop_activity_{idx}.gif", writer="imagemagick", fps=10)
    except:
        print(f"Laptop data for {idx} not available")

# %%
