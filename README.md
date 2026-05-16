# Implicit Hate Speech Against Swiss Minorities

This project, realized in the scope of the EE-559 course at EPFL, is dedicated to exploring hate speech recognition based on the [TOXIGEN](https://github.com/microsoft/ToxiGen) dataset, extended to include swiss minority groups.

**This repository contains and discusses content that is offensive or upsetting. All materials are intended to support research that improves toxicity detection methods. Included examples of toxicity do not represent how the authors feel about any identity groups.**

| Full name                | SCIPER                                   |
|:-------------------------|:-----------------------------------------|
| Louis Grange             | [341237](https://people.epfl.ch/341237)  |
| Oussama Yazidi           | [311471](https://people.epfl.ch/311471)  |
| Joshua Oyewole Oyebanji  | [347485](https://people.epfl.ch/347485)  |

## Dataset generation

- [x] Writen offensive examples of toxicity based on the TOXIGEN format, saved as `prompts/*_sentences.txt` files
- [x] Formatted these sentences as different combinations of prompts in `prompts/*_1k.txt` files using `sentences.py`
- [ ] Generated 1000 samples from each of these files in `data/*_1k.txt` to use for training/benchmarking using `dataset.py` and OpenAI Platform
- [ ] Benchmarked the performance of the TOXIGEN model `tomh/toxigen_roberta` on the generated dataset
- [ ] Trained a new network on the extended dataset with swiss minorities
- [ ] Benchmarked the performance of the extended model on the whole dataset