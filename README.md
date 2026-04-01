# GVGAI_jpype — GVGAI Benchmark for LLMs

This project is an ongoing effort to create a **benchmark for Large Language Models (LLMs)** within the **Generic Video Game AI (GVGAI)** framework. It is a continuation and rewrite of [doveliyuchen/GVGAI_GYM](https://github.com/doveliyuchen/GVGAI_GYM), which was itself forked from [rubenrtorrado/GVGAI_GYM](https://github.com/rubenrtorrado/GVGAI_GYM).

The key change in this rewrite is the replacement of the original Java socket-based communication with **JPype**, enabling direct in-process JVM integration from Python. This eliminates the socket overhead, simplifies setup, and improves compatibility with Python 3.11+.

The GVGAI framework has been widely used since 2019 for competitions like the **GVGAI Learning Competition** and research in reinforcement learning and planning agents. While traditional approaches relied on access to forward models or reinforcement learning, this project explores how LLMs can interact with game states—both text-based and image-based—to make decisions in these environments.

For more details about the original competition rules, rankings, and related resources, visit the [AI in Games website](http://www.aingames.cn/), maintained by [Hao Tong](https://github.com/HawkTom) and [Jialin Liu](https://github.com/ljialin).

---

## Key Features

- **LLM Integration**: The framework now supports sending **text-based states** and **image-based states** to LLMs, enabling them to interpret and act within GVGAI games.
- **Benchmarking**: Evaluate LLM performance across various GVGAI game levels, comparing their decision-making capabilities to traditional planning and RL agents.
- **Ongoing Development**: This project is actively being developed to refine the benchmark and explore new ways to integrate LLMs into game AI research.

---


## Installation (New)

This project now uses **JPype + modern Python/Java versions**. The old Docker/JDK9 installation flow has been removed.

### 1) Create environment from `environment.yml`

```bash
conda env create -f environment.yml
conda activate gvgai_jpype
```

### 2) Install this package in editable mode

```bash
pip install -e .
```

`setup.py` will compile the Java sources during install/develop.

---

## Environment Requirements

- **OS**: Windows / Linux / macOS (with Conda support)
- **Conda**: Miniconda or Anaconda
- **Python**: 3.11
- **Java**: OpenJDK 17
- **Core dependencies**:
  - gymnasium
  - JPype1
  - numpy / pillow / matplotlib
  - openai / anthropic / google-generativeai
  - pandas / flask / tqdm

All exact dependency definitions are maintained in:
- `environment.yml`
- `requirements.txt`

---

## Resources

[GVGAI website](http://www.gvgai.net)

[original GVGAI-Gym (master branch)](https://github.com/rubenrtorrado/GVGAI_GYM) 

[Demo video on YouTube](https://youtu.be/O84KgRt6AJI)

[AI in Games website for more about competition updates](http://www.aingames.cn/#sources)

[*Deep Reinforcement Learning for General Video Game AI*](https://arxiv.org/abs/1806.02448) published at IEEE CIG2018

---
## References

1. [G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” 2016.](https://github.com/openai/gym)
2. [A. Hill, A. Raffin, M. Ernestus, A. Gleave, A. Kanervisto, R. Traore, P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu, “Stable baselines,” https://github.com/hill-a/stable-baselines, 2018.](https://github.com/hill-a/stable-baselines)
3. [R. R. Torrado, P. Bontrager, J. Togelius, J. Liu, and D. Perez-Liebana, “Deep reinforcement learning for general video game AI,” in Computational Intelligence and Games (CIG), 2018 IEEE Conference on. IEEE, 2018.](https://github.com/rubenrtorrado/GVGAI_GYM)
4. [D Perez-Liebana, J Liu, A Khalifa, RD Gaina, J Togelius, SM Lucas, "General video game AI: A multitrack framework for evaluating agents, games, and content generation algorithms," in IEEE Transactions on Games, 11(3), 195-214.](https://arxiv.org/pdf/1802.10363)
