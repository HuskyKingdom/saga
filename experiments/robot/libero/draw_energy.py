import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('ours.pkl', 'rb') as f:
    ours = pickle.load(f)


with open('baseline.pkl', 'rb') as f:
    baseline = pickle.load(f)


E_base = baseline
E_hnn = ours

dE_base = E_base - E_base[0]
dE_hnn  = E_hnn  - E_hnn[0]

ts = np.arange(len(dE_base)/2)
plt.figure()
plt.plot(ts, dE_base[:8], label='Baseline ΔE')
plt.plot(ts, dE_hnn[:8],  label='With HNN ΔE')
plt.xlabel('Timestep')
plt.ylabel('ΔDynamic Energy')
plt.legend()
plt.savefig('/mnt/nfs/sgyson10/openvla-oft-yhs/dynamic_energy_drift.pdf')
plt.close()

print("saved")