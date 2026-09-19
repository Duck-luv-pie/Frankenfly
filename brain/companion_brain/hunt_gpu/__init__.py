"""The hunt arena on a GPU: thousands of flies hunting walking humans at once, trained by PPO.

A spinoff of `sim/hunt.py` (which trains the connectome brain on all CPU cores, a few flies at a
time). Here the arena is a batch of tensors (`arena.py`, the same physics, seeds and motor
conventions as `sim/hunt_arena.py`), the fly is a compact recurrent network shaped like the fly's
hunting circuit (`brain.py`: hot cells -> Kenyon-cell expansion -> MBON valence; retinotopic LC
columns -> pursuit; a central-complex GRU; descending forward / turn drive), and the trainer is
recurrent PPO (`ppo.py`). `bridge.py` drives the CPU arena with a trained network so the two arenas
can be checked against each other. Needs the `gpu` extra: `uv sync --extra gpu`."""
