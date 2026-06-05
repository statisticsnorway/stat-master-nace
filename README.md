# stat-master-nace

This code is created by Minauio <minash19@yahoo.com>, and is a part of a Master’s thesis submitted to the University of Oslo, conducted in collaboration with Statistics Norway (SSB). The task is to compare the automation of Norwegian NACE classification (SIC) using hierarchical and flat prediction methods with FastText and large language models. This repository contains the code used for the experiments reported in the master's thesis. The repository is intended to facilitate reproducibility of the reported results.

## Supervised Models

**Dummy baseline model:** Implemented and evaluated in `run_dummy.py`.

**FastText flat model:** Implemented and evaluated in `run_flat_fastxt.py`.

**FastText hierarchical model:** Implemented and evaluated in `run_fasttxt_hier.py`.

## LLM Models

**Flat approach with different configurations:** Implemented and evaluated in `llm_classifier_flat_config.py`.

**Flat approach:** Implemented and evaluated in `llm_classifier_flat.py`.

**Hierarchical approach:** Implemented and evaluated in `llm_classifier_hier.py`.

## Utilities

### Supervised Models

**FastText flat model:** Implemented in `src/utils/baseline_utils.py`.

**FastText hierarchical model:** Implemented in `src/utils/baseline_hier.py`.

### LLM Models

**Flat approach:** Implemented in `src/utils/llm/flat_llm_util.py`.

**Hierarchical approach:** Implemented in `src/utils/llm/hier_llm_util.py`.


## Evaluation

Evaluation metrics and the per-level evaluation are implemented in `src/metrics.py`.

## Preprocessing of the data

The data is preprocessed the FastText and llm methods, then split into train, validation and test sets. These processes are performed in `src/preprocess.py`.
