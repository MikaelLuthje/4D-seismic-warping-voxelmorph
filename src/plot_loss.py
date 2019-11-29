#%%
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("../miccai_log.csv")
df["minutes"] = (df.epoch+1)*5

fig, ax = plt.subplots(figsize=(7,5))
df.drop(["epoch","minutes"], axis=1).plot(figsize=(7,5), ax=ax)
ax.legend(["KL Divergence", "Total Loss", "Data Loss"])
ax.set_title('Voxelmorph Losses')
ax.set_xlabel('Time [m]')
ax.set_ylabel('Loss')
ax.set_ylim(0,70)
fig.savefig("miccai_loss.png", bbox_inches='tight')

#%%
