import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

log_dir_ppo = "../runs/195/PPO_1" 
log_dir_a2c = "../runs/215/A2C_1" 

def get_df(log_dir):
    reader = SummaryReader(log_dir)
    df_scalars = reader.scalars
    df_rewards = df_scalars[
        df_scalars['tag'].str.contains('rew|reward', case=False, na=False)
    ].copy()
    df_rewards_pivot = df_rewards.pivot(
        index='step', 
        columns='tag', 
        values='value'
    ).reset_index() 
    return df_rewards_pivot

df_rewards_pivot_ppo = get_df(log_dir_ppo)
df_rewards_pivot_a2c = get_df(log_dir_a2c)

# Calcola la media smoothata con filtro gaussiano
ppo_smoothed = gaussian_filter1d(df_rewards_pivot_ppo["rollout/ep_rew_mean"], sigma=5.0)
a2c_smoothed = gaussian_filter1d(df_rewards_pivot_a2c["rollout/ep_rew_mean"], sigma=5.0)

# Plot corretto con le label giuste
fig, ax = plt.subplots(figsize=(12, 6))

# Plot dati grezzi (trasparenti)
ax.plot(df_rewards_pivot_ppo["step"], df_rewards_pivot_ppo["rollout/ep_rew_mean"], 
        linewidth=1, alpha=0.3, color='tab:blue')
ax.plot(df_rewards_pivot_a2c["step"], df_rewards_pivot_a2c["rollout/ep_rew_mean"], 
        linewidth=1, alpha=0.3, color='tab:orange')

# Plot dati smoothati (opachi)
ax.plot(df_rewards_pivot_ppo["step"], ppo_smoothed, linewidth=2, label="PPO (smoothed)", color='tab:blue')
ax.plot(df_rewards_pivot_a2c["step"], a2c_smoothed, linewidth=2, label="A2C (smoothed)", color='tab:orange')

ax.set_xlabel("Episode (Step)")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Training Mean Reward over Time")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("reward.png", dpi=150, bbox_inches='tight')
plt.savefig("reward.pdf", dpi=150, bbox_inches='tight')

plt.close()
