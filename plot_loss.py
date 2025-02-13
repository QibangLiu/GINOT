# %%
import json
import matplotlib.pyplot as plt

# %%
file = "/work/hdd/bdsy/qibang/repository_Wbdsy/GINOT/models/saved_weights/JEB_geo_from_pc_test/logs.json"
with open(file, 'r') as f:
  data = json.load(f)
# %%
plt.plot(data['loss'], label='train')
plt.plot(data['val_loss'], label='val')
plt.yscale('log')
plt.legend()
# %%
