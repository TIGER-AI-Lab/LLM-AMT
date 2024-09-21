# LLM-AMT: Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering

This repository contains the code for our paper "Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering", which has been accepted as a Findings paper at EMNLP 2024.

## Status

We are currently in the process of organizing and cleaning up the code for LLM-AMT. The repository will be updated incrementally as we prepare different components of the project for public release.

## Updates

- [2024-09-22]: Initial repository setup
- [Future dates]: Upcoming code releases (TBA)

## Abstract

Large Language Models (LLMs) like ChatGPT have demonstrated impressive abilities in generating responses based on human instructions. However, their use in the medical field can be challenging due to their lack of specific, in-depth knowledge.

In this study, we present a system called LLMs Augmented with Medical Textbooks (LLM-AMT) designed to enhance the proficiency of LLMs in specialized domains. LLM-AMT integrates authoritative medical textbooks into the LLMs' framework using plug-and-play modules. These modules include:

- A *Query Augmenter*
- A *Hybrid Textbook Retriever*
- A *Knowledge Self-Refiner*

Together, they incorporate authoritative medical knowledge. Additionally, an *LLM Reader* aids in contextual understanding.

Our experimental results on three medical QA tasks demonstrate that LLM-AMT significantly improves response quality, with accuracy gains ranging from 11.6% to 16.6%. Notably, with GPT-4-Turbo as the base model, LLM-AMT outperforms the specialized Med-PaLM 2 model pre-trained on a massive amount of medical corpus by 2-3%.

We found that despite being 100× smaller in size, medical textbooks as a retrieval corpus are proven to be a more effective knowledge database than Wikipedia in the medical domain, boosting performance by 7.8%-13.7%.

## Citation

If you use this code or find our work helpful, please consider citing our paper:

```bibtex
@article{wang2023augmenting,
  title={Augmenting black-box llms with medical textbooks for clinical question answering},
  author={Wang, Yubo and Ma, Xueguang and Chen, Wenhu},
  journal={arXiv preprint arXiv:2309.02233},
  year={2023}
}