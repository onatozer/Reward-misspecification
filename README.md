## Installation:
After cloning the repo, run:

Doing this because I wanted to use stable-baselines instead of having to re-create all the RL algorithms from scratch, but the safe-grid-world gym environment breaks on newer versions of numpy.
```bash
pip install "numpy<2.0"
pip install git+https://github.com/david-lindner/safe-grid-gym.git
PYCOLAB_FILE=$(python -c "import pycolab.ascii_art; print(pycolab.ascii_art.__file__)")
sed -i 's/art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in art)/art = np.vstack([np.frombuffer(line.encode(), dtype=np.uint8) for line in art])/' "$PYCOLAB_FILE"
pip install "stable-baselines3<2.0"
pip install torch cloudpickle pandas matplotlib
pip install -r requirements.txt
git clone https://github.com/david-lindner/safe-grid-gym.git

```

TODO
- [ ] Fix the git submodule stuff so that changes to cloned repos are properly tracked