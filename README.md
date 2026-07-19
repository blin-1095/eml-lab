# Setup

This project is managed using **uv**.
You need to first install **uv** to be able to initialize the python environment.

After installing uv execute

```
uv sync
```

After that you can source the venv as usual and execute the scripts.



# General notes

## Efficiency score

To select the best performing model after pruning, we have to take into account the Loss and the achieved performance gain.
Loss alone is not enough as the metric, since the lowest Loss model will always be the unpruned one (or models that have been pruned less).

To get the best performing model per FPS, we use the following equation:

$$
\text{Efficiency Score} = \frac{\text{Validation Loss}}{\left( \frac{\text{FPS}_{\text{current}}}{\text{FPS}_{\text{baseline}}} \right)}
$$

This calculates an Efficiency Score which allows us to compare whether the rise in Loss is worth the speedup gain from pruning.

