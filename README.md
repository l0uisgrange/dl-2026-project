# Implicit Hate Speech Against Swiss Minorities

This project, realized in the scope of the EE-559 course at EPFL, is dedicated to exploring hate speech recognition based on the [TOXIGEN](https://github.com/microsoft/ToxiGen) dataset, extended to include swiss minority groups.

**This repository contains and discusses content that is offensive or upsetting. All materials are intended to support research that improves toxicity detection methods. Included examples of toxicity do not represent how the authors feel about any identity groups.**

| Full name                | SCIPER                                   |
|:-------------------------|:-----------------------------------------|
| Louis Grange             | [341237](https://people.epfl.ch/341237)  |
| Oussama Yazidi           | [311471](https://people.epfl.ch/311471)  |
| Joshua Oyewole Oyebanji  | [347485](https://people.epfl.ch/347485)  |

## Process overview

- [x] Writen offensive examples of toxicity based on the TOXIGEN format, saved as `prompts/*_sentences.txt` files
- [x] Generated 1000 samples from each of these files in `data/*.txt` to use for training/benchmarking using `gpt-4o-mini` on OpenAI Platform
- [ ] Benchmarked the performance of the TOXIGEN model `tomh/toxigen_roberta` on the generated dataset
- [ ] Trained a new network on the extended dataset with swiss minorities
- [ ] Benchmarked the performance of the extended model on the whole dataset

## Dataset

Each minority group (asylum seekers, portuguese, cross-border workers) are represented by a CSV file in the `data` folder, containing

- **`text`**: generated sample corresponding to the ouput of `gpt-4o-mini` from the sentences in the `prompts` folder
- **`mode`**: 1 meaning the sample was generated from a `hate` dataset and 0 being generated from the `neutral` one
- **`target_group`**: targeted group (seekers, portuguese, cross-border workers)
- **`toxicity_ai`**: ai annotated toxicity score on a scale from 1 to 5 (1 being benign and 5 being clearly offensive)
- **`toxicity_human`** (TODO): ai annotated toxicity score on a scale from 1 to 5 (1 being benign and 5 being clearly offensive)
