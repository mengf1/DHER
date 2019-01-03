# Dy Envs
There are dynamic goal environments. We modify the robotic manipulation environments created
by OpenAI (Brockman et al., 2016) for our experiments. 

![optional caption text](resource/fig-tasks.jpeg)

As shown in above figure, we assign certain rules to the goals so that they accordingly move in the environments while an agent is required to control the robotic arm's grippers to reach the goal that moves along a straight line (Dy-Reaching), to reach the goal that moves in a circle (Dy-Circling), or to push a block to the goal that moves along a straight line (Dy-Pushing). 

## How to install it

Our environments depend on [openai gym](https://github.com/openai/gym). Please install gym at first.

``` shell
cd dygym
python install -e .
```

## Test new environments

``` shell
cd dygym/test
python test_dyreach.py
```

# DHER
Update soon ~
