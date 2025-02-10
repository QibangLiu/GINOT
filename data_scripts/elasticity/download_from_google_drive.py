# %%
import gdown
import numpy as np
import matplotlib.pyplot as plt
# %%
# elasticity


def download_data(data_path):
    # Download data from google drive:
    # https://drive.google.com/drive/folders/1f9i6eOHfjULfVbd2FbUHHjqcL6CPJxbb
    file_id = ["1I-fO-RsFvD3nqBuFrg67R0yqTFdD_gpA",
               "18MsdeVVrmQacARYzWg8mXqEOFBTrhCVd", "1Ia5izgUum-IQLdO6PW70HO8AdAqA_IVb", "1Pjliqhxegoe5VpoLrpBa9n3P4gX9MfTt"
               ]  # Extract from the shared URL
    for file in file_id:
        url = f"https://drive.google.com/uc?id={file}"
        gdown.download(url, data_path,)


data_path = "../../data/elasticity/"

# %%
# check the data
XY = np.load(data_path+"Random_UnitCell_XY_10.npy")
rr = np.load(data_path+"Random_UnitCell_rr_10.npy")
sigma = np.load(data_path+"Random_UnitCell_sigma_10.npy")
theta = np.linspace(0, 2*np.pi, len(rr))[:, None]
pc = np.array([rr*np.cos(theta), rr*np.sin(theta)]).T+0.5
pc = pc[:, :-1, :]

# Find boundary points from XY
x_q = np.linspace(0, 1, 16)
y_q = np.linspace(0, 1, 16)
top_row = np.column_stack((x_q, np.full_like(x_q, 0)))
bottom_row = np.column_stack((x_q, np.full_like(x_q, 1)))
left_column = np.column_stack((np.full_like(y_q, 0), y_q))
right_column = np.column_stack((np.full_like(y_q, 1), y_q))
boundary_points = np.vstack((top_row, bottom_row, left_column, right_column))
boundary_points = np.tile(boundary_points[None], (len(pc), 1, 1))

pc = np.concatenate([pc, boundary_points], axis=1)

data = {"nodes": XY.transpose(
    2, 0, 1), "sigma": sigma.transpose(1, 0), "points_cloud": pc}

# Save data to npz file
np.savez(data_path + "elasticity.npz", **data)
# %%
