# Snake reinforcement-learning lab

This environment is both a playable visualization and a small comparison lab.
Every algorithm receives the same 11 binary features and chooses one of three
relative actions: go straight, turn right, or turn left.

## Algorithms

| CLI name | Family | What changes |
| --- | --- | --- |
| `dqn` | Deep RL | MLP plus a target network; the target network selects and evaluates the maximum action. |
| `double_dqn` | Deep RL | The online network selects the next action and the target network evaluates it, reducing maximization bias. |
| `dueling_dqn` | Deep RL | Separate state-value and action-advantage heads with the standard DQN target. |
| `dueling_double_dqn` | Deep RL | Dueling heads with Double-DQN target selection. |
| `q_learning` | Tabular RL | An inspectable Q table over the `2^11` possible observations. |
| `sarsa` | Tabular RL | Expected SARSA: the bootstrap is the epsilon-greedy policy's expected value. It learns online and deliberately does not replay old transitions. |

The neural models use light dropout, Adam weight decay, gradient clipping, a
bounded replay buffer, and a periodically synchronized target network. Existing
two-layer and three-layer MLP checkpoints remain loadable by `dqn` and
`double_dqn`. New checkpoint metadata records the algorithm so an incompatible
dueling network or Q table is never loaded accidentally.

## Generalization instead of memorization

Training, validation, and final testing use deterministic but independent seed
streams. Each training episode gets a new seed. Domain randomization changes the
snake's valid starting cell and heading, while the curriculum increases that
variation in three stages: `orientation`, `spawn_shift`, and `generalization`.

Every 25 games, by default, the current greedy policy runs on the same fixed
eight validation seeds. Keeping that suite unchanged makes checkpoint means
directly comparable. A separately rooted fixed 20-episode final-test suite is
never used for model selection. The inspector and terminal show validation mean,
standard deviation, and the generalization gap:

```text
generalization gap = rolling training mean - validation mean
```

A large positive gap is a warning that training performance is not transferring.
Validation and final testing never train, write replay memory, consume the
exploration RNG, or add unseen states to a tabular policy.

Checkpoints are written when the validation mean improves, a new training record
is reached, or the tracked training mean improves by more than 5%. Once
validation-backed checkpoints exist, loading is validation driven and prefers
the best validation mean; unevaluated training-only candidates no longer outrank
them. Legacy checkpoint sets without evaluation metadata still fall back to
training score.

New metadata records the environment, train/validation/final-test roots, exact
validation and final-test seed lists, curriculum settings, and validation round.
`--eval-only` reuses that stored validation suite when the environment matches and
no seed/count override is supplied, making the recorded validation mean
reproducible. Use `--eval-suite final_test` only for the final report.

Resuming restores the learned network weights or Q table and episode metadata,
but not optimizer state, replay contents, learner RNG state, or target-sync
counters. Continued training is therefore not bit-for-bit deterministic.
Evaluation reproduction remains exact because it needs only the frozen policy
and stored suite.

## Educational environments

| Preset | Size | Learning focus |
| --- | ---: | --- |
| `tutorial` | 200 × 160 | Fast feedback on relative actions and danger features. |
| `compact` | 320 × 240 | Tighter walls and shorter planning routes. |
| `standard` | 640 × 480 | Default training and evaluation. |
| `wide` | 720 × 360 | Transfer to a different aspect ratio. |

Run a short reproducible comparison from the repository root:

```bash
python -m snakeGameQDlearning.main \
  --algorithm dqn --games 100 --headless --fresh --no-save --seed 7
python -m snakeGameQDlearning.main \
  --algorithm double_dqn --games 100 --headless --fresh --no-save --seed 7
python -m snakeGameQDlearning.main \
  --algorithm dueling_double_dqn --games 100 --headless --fresh --no-save --seed 7
python -m snakeGameQDlearning.main \
  --algorithm q_learning --games 100 --headless --fresh --no-save --seed 7
python -m snakeGameQDlearning.main \
  --algorithm sarsa --games 100 --headless --fresh --no-save --seed 7
```

Keep `--seed`, `--evaluation-seed`, `--final-test-seed`, `--environment`,
`--eval-every`, and episode counts identical for a controlled comparison.
`--environment` is also part of checkpoint compatibility: the current CLI loads
only checkpoints trained for the selected preset. Cross-preset transfer therefore
requires loading the policy through the Python API. To evaluate an existing
checkpoint from the `wide` family:

```bash
python -m snakeGameQDlearning.main --algorithm dueling_double_dqn \
  --eval-only --eval-suite final_test --environment wide \
  --final-test-episodes 20 --final-test-seed 12001
```

Useful experiment controls:

```text
--no-domain-randomization   keep the classic centered, right-facing spawn
--no-curriculum             enable full spawn randomization immediately
--eval-every 0              disable periodic validation
--eval-suite final_test     use the reserved suite in --eval-only mode
--fresh                     start without loading a compatible checkpoint
--no-save                   keep comparison runs from writing checkpoints
```

The 11 features are three immediate collision flags, four one-hot heading
flags, and four food-direction flags. This compact representation makes the
tabular algorithms possible, but it is partially observable: it cannot describe
the entire body layout. Comparing the methods is therefore an exercise in both
RL algorithms and representation limits.
