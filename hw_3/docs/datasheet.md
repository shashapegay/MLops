# Datasheet — OpenOrca v1/v2

## Source

- Dataset: `Open-Orca/OpenOrca`
- URL: https://huggingface.co/datasets/Open-Orca/OpenOrca
- Split: `train`
- Sampling: streaming
- Selection size: exactly 5000 source records before cleaning
- Stratification key: `system_prompt`

## Source schema

The source records contain:

- `id`
- `system_prompt`
- `question`
- `response`

Mapping:

- `id` -> `id`
- `system_prompt` -> `messages[0]`
- `question` -> `messages[1]`
- `response` -> `messages[2]`

## Sampling

The complete source is scanned once to count every
`system_prompt` stratum.

Target counts are allocated proportionally
using the largest-remainder method.

The sum of target counts is exactly 5000.

A second streaming pass selects records within
each stratum using a deterministic SHA-256 priority.

v1 uses seed 42.

v2 uses seed 2026.

## Cleaning

1. schema validation;
2. length filtering;
3. PII masking;
4. exact deduplication;
5. near-duplicate deduplication.

Near-duplicate connected components are then used
as split groups.

## Split

The cleaned dataset is divided approximately:

- train: 80%;
- validation: 10%;
- test: 10%.

Groups are never intentionally divided between
train and test.

## Contamination

The following intersections are checked:

- id;
- normalized question;
- group;
- near-duplicate pairs.

All four must be zero.

## Diversity

The dataset must satisfy the configured:

- minimum number of examples;
- minimum number of system prompts;
- minimum number of groups;
- maximum group share;
- answer-length diversity;
- duplicate-answer share.

Any violation fails the pipeline.

## Limitations

OpenOrca does not contain a native `topic` field.

The homework-compatible `topic` field is therefore used
as a technical group identifier after near-duplicate clustering.

The selected 5000 records are a sample of the complete source.