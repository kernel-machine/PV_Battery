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

tsp -L "best latent quant" python -u main.py --use_solar --use_hour_minute --use_images --use_quantize_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 128 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 32
tsp -L "best latent quant" python -u main.py --use_solar --use_hour_minute --use_images --use_quantize_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 256 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 32
tsp -L "best latent quant" python -u main.py --use_solar --use_hour_minute --use_images --use_quantize_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 256 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 64
tsp -L "best latent quant" python -u main.py --use_solar --use_hour_minute --use_images --use_quantize_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 256 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 128
tsp -L "best latent quant" python -u main.py --use_solar --use_hour_minute --use_images --use_quantize_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 512 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 128

tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 128 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 24
tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 128 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 32
tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 128 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 48
tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 512 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 48
tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 512 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 64
tsp -L "best latent embed" python -u main.py --use_solar --use_hour_minute --use_images --use_embed_prev_day --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 128 --lr_decay lin --train_days 300 --autostart --start_thr 0.05 --latent_size 64