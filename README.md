# HOW TO RUN

- Put [solcast2024.csv](https://github.com/user-attachments/files/21213151/solcast2024.csv) and [solcast2025.csv](https://github.com/user-attachments/files/21213140/solcast2025.csv) on the project folder (not in SRC) but in the main folder
- install the python libraries in the `requirements.txt`

```bash
cd src
python -u main.py --discrete_action --alg ppo --n_env 8 --batch_size 512 --incentive_factor 0.7
```

These are the best parameters find untill now

## Important files
- **main.py** contains the training and evaluation code
- **lib/env.py** contains the RL Environment
- **lib/device.py** contains the code to simulate the device
