# Project context

Evaluate whether the Assistant Axis and its reported safety benefits generalize from synthetic evaluations to real, multi-turn conversations such as WildChat.

## Motivation

The [Assistant Axis paper](https://github.com/safety-research/assistant-axis) describes an activation direction associated with Assistant-like behavior. The user's proposal highlights two reported safety claims: movement away from this direction predicts harmful behavior, and activation capping reduces persona-based jailbreaks. This project tests whether those relationships hold in natural conversations.

Synthetic extraction and evaluation may share artifacts, and generated prompts may express traits more explicitly than real chats. The user's motivating reference is Thomas Jiralerspong's evaluation of Persona Vectors by Chen et al., 2025. That evaluation reportedly found that the relationship between activation projections and expressed traits seen on synthetic data disappeared on WildChat. The concern is that the Assistant Axis may face similar limits.

## Persona drift

Study how the model's position along the Assistant Axis changes over real conversations. Determine whether natural interactions produce sustained movement away from Assistant-like behavior and whether the axis meaningfully tracks that change. The goal is to understand how well the paper's account of persona drift describes real conversational dynamics.

## Harmful behaviors

Test whether movement away from the Assistant Axis predicts harmful responses after real conversation histories. Establish whether the reported relationship between drift and harm holds beyond synthetic persona prompts. A lower projection alone should not be treated as evidence of harmful behavior.

## Activation capping

Evaluate whether the authors' activation-capping intervention reduces harmful behavior in real conversational settings. Assess its effect on ordinary assistance and useful capabilities as well as safety. The goal is to understand whether the reported safety benefits transfer and what tradeoffs arise in practice.

These are empirical questions, not assumed outcomes. Treat literature claims and proposed experiments as research context. Distinguish implemented methods, completed runs, and observed results when describing progress or drawing conclusions.
