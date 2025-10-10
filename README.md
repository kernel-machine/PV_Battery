# HOW TO RUN

- Put [solcast2024.csv](https://github.com/user-attachments/files/21213151/solcast2024.csv) and [solcast2025.csv](https://github.com/user-attachments/files/21213140/solcast2025.csv) on the project folder (not in SRC) but in the main folder
- install the python libraries from the `requirements.txt`
- Run the `main.py` from the `src` directory
    ```bash
    cd src
    python -u main.py --discrete_action --alg ppo --n_env 8 --batch_size 512 --incentive_factor 0.7
    ```
    These are the best parameters find untill now

## Important files
- **main.py** contains the training and evaluation code
- **lib/env.py** contains the RL Environment
- **lib/device.py** contains the code to simulate the device


BEST
python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --epochs 10
197

tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 0.5
tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 0.6
tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 0.7
tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 0.8
tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 0.9
tsp python -u main.py --use_solar --use_hour --use_month --steps 3000000 --lr 0.0001 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --forecast_minutes 60 --update_steps 50 --prediction_accuracy 1

python -u main.py --use_solar --use_hour --use_month --steps 50000000 --lr 0.0007 --n_env 1 --term_days 7 --alg a2c --use_estimate_forecast --choose_forecast --update_steps 50 --prediction_accuracy 0.7