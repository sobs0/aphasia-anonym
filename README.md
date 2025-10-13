# aphasia-anonym
The code implementation for the study "Towards Privacy-Preserving Fine-Tuning: Aphasic Speech Anonymization for Effective ASR".

All pipelines can be executed by configuring the config.yaml and running the master.py script. All functions are located in the corresponding directrory.

## Reproducibility
The fine-tuning of the wav2vec2.0 models was conducted on an NVIDIA GPU RTX 6000 Ada with 49GB VRAM and AMD CPUs EPYC 9354 with a total system capacity of of 1.5TB RAM.
The experiment was implementated in Python (version 3.10.12) [1], using the PyTorch based Training API of the Hugging Face Transformers framework for model fine-tuning (version 4.55.2) [2]. Statistical analysis was done in R (version 4.4.2) [3].


[1]: Python Software Foundation. 2022. The Python Language Reference.
[2]: Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Perric Cistac, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. Association for Computational Linguistics.
[3]: R Core-Team. 2024. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing.
