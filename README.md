## Installation:
After cloning the repo, run:

Doing this because I wanted to use stable-baselines instead of having to re-create all the RL algorithms from scratch, but the safe-grid-world gym environment breaks on newer versions of numpy.
```bash
git clone https://github.com/timorl/ai-safety-gridworlds/tarball/master#egg=ai-safety-gridworlds-1.5
pip install "numpy<2.0"
pip install "stable-baselines3<2.0"
pip install torch cloudpickle pandas matplotlib
pip install -r requirements.txt
```
