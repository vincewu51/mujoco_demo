# IsaacLab Setup Instructions

**Date:** Sep 4, 2025

---

## Run Example

```bash
xvfb-run -a python get_started/0_static_scene.py --sim mujoco --headless
```

---

## Step 1: Install pyenv

```bash
curl https://pyenv.run | bash
```

Add the following lines to your shell config file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Apply the changes:

```bash
source ~/.bashrc    # or source ~/.zshrc
```

---

## Step 2: Install Python 3.10 via pyenv

```bash
pyenv install 3.10.15
```

---

## Step 3: Use Python 3.10

```bash
pyenv shell 3.10.15
```

---

## Step 4: Set up SSH Keys and Clone IsaacLab

```bash
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
git clone --depth 1 --branch v1.4.1 git@github.com:isaac-sim/IsaacLab.git IsaacLab141 && \
cd IsaacLab141 && \
./isaaclab.sh -i none
```
