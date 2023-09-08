#!/bin/bash

# Launch Jupyter notebook on the remote server
ssh -i C:\Users\kolon\Documents\MI-Project\.ssh\id_ed25519 user80@141.13.17.218 'jupyter notebook --no-browser --port=8889' &

# Launch TensorBoard on the remote server
ssh user80@141.13.17.218 'tensorboard --logdir /path/to/your/logs --port=6006' &

# Forward the Jupyter notebook and TensorBoard ports
ssh -N -f -L localhost:8888:localhost:8889 user80@141.13.17.218 &
ssh -N -f -L localhost:6006:localhost:6006 user80@141.13.17.218 &
