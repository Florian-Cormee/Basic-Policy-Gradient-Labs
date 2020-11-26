rm -rf data/
python3 main.py \
  --env_name Pendulum-v0 \
  --env_obs_space_name cos_th theta_dt \
  --study_name pg \
  --policy_type sac \
  --gradients discount \
  --critic_estim_method td \
  --team_name beroujon-cormee \
  --nb_repet 1 \
  --nb_trajs 800 \
  --max_episode_steps 200 \
  --update_threshold 1000 \
  --nb_updates 20 \
  --batch_size 256 \
  --print_interval 20 \
  --gamma 0.98 \
  --tau 0.01 \
  --lr_actor 0.0005 \
  --lr_critic 0.001 \
  --init_alpha 0.02 \
  --lr_alpha 0.0
