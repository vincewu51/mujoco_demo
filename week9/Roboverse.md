Sep 4, 2025
xvfb-run -a python get_started/0_static_scene.py  --sim mujoco --headless


# 1️⃣ Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Add pyenv to your shell
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 2️⃣ Install Python 3.10 via pyenv
pyenv install 3.10.15

# 3️⃣ Create a Python 3.10 venv using pyenv’s Python
pyenv shell 3.10.15   # temporarily switch to Python 3.10

mkdir -p /root/.ssh && \
cp /workspace/.ssh/id_ed25519 /root/.ssh/ && \
cp /workspace/.ssh/id_ed25519.pub /root/.ssh/ && \
cp /workspace/.ssh/known_hosts /root/.ssh/ 2>/dev/null || true && \
chmod 700 /root/.ssh && \
chmod 600 /root/.ssh/id_ed25519 && \
chmod 644 /root/.ssh/id_ed25519.pub && \
chmod 644 /root/.ssh/known_hosts 2>/dev/null || true && \
eval "$(ssh-agent -s)" && \
ssh-add /root/.ssh/id_ed25519 && \
ssh -o StrictHostKeyChecking=no -T git@github.com && \
git clone --depth 1 --branch v1.4.1 git@github.com:isaac-sim/IsaacLab.git IsaacLab141 && cd IsaacLab141 && \
./isaaclab.sh -i none


