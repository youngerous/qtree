<div align="center">
    
# Learning to Explore and Select for Coverage-Conditioned Retrieval-Augmented Generation

[![Paper](https://img.shields.io/badge/Paper-arxiv.2407.01158-red)](https://arxiv.org/abs/2407.01158)

#### Takyoung Kim<sup>1,&ast;</sup>, Kyungjae Lee<sup>2</sup>, Young Rok Jang<sup>2</sup>, Ji Yong Cho<sup>2,3</sup>, Gangwoo Kim<sup>4,&ast;</sup>, Minseok Cho<sup>2</sup>, Moontae Lee<sup>2,5</sup> <br> <sub><sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>LG AI Research, <sup>3</sup>Cornell University, <sup>4</sup>Korea University, <sup>5</sup>University of Illinois Chicago</sub> <br> <sub><sup>&ast;</sup>Work done as a research intern at LG AI Research</sub>

</div>

## Abstract

> Interactions with large language models (LLMs) often yield long and detailed responses, leveraging both parametric knowledge and retrieval-augmented generation (RAG). While these responses can provide rich insights, they often include redundant or less engaging content not aligned with user interests. This issue becomes apparent when users specify particular subtopics to include or exclude -- termed coverage-conditioned ($C^2$) queries -- as LLMs often struggle to provide tailored responses. To address this challenge, we investigate the role of query outlines, sequences of subqueries designed to guide LLMs in generating responses that meet specific user requirements. To systematically create and evaluate these outlines, we introduce **QTree**, a dataset of 10K hierarchical sets of information-seeking subqueries that define structured boundaries for outline creation and evaluation in $C^2$ scenarios. Additionally, we develop **QPlanner**, a 7B language model trained to generate customized outlines within boundaries of QTree. We evaluate the effectiveness of the generated outlines through automatic and human judgements, focusing on their impact within retrieval-augmented generation (RAG) systems. Experimental results demonstrate that QPlanner, especially when trained with alignment techniques like DPO, generates higher-quality outlines that better fulfill diverse user needs. 

## Resource (QTree)

### Train set
- \# of dataset: 10,580 [[LINK](https://drive.google.com/file/d/1CIv0oTusKvRuJZwaR7x5aRrFvBnC3hdt/view?usp=share_link)]
    - Note: There are three more samples than those specified in the paper.
- Configuration
    - ```question```: Base query ($q_{base}$)
    - ```instruction```: Coverage query ($q_{cov}$)
    - ```background```: Background query
    - ```intention```: Intent operation (include/exclude)
    - ```tree```: QTree (a hierarchical set of queries)
    - ```candidates```: Three candidate query outlines (i.e., four subqueries from QTree) extracted by LLM

### Test set
- \# of dataset: 300 [[LINK](https://drive.google.com/file/d/1sVrIb7iMZDaq7ZvgVdHMcr56wxFVxUpk/view?usp=share_link)]
- Configuration
    - ```question```: Base query ($q_{base}$)
    - ```instruction```: Coverage query ($q_{cov}$)
    - ```background```: Background query
    - ```intention```: Intent operation (include/exclude)
    - ```tree```: QTree (a hierarchical set of queries)

## Acknowledgement
- Our QTree is based on seed queries from [ASQA](https://arxiv.org/abs/2204.06092), [Longform](https://arxiv.org/abs/2304.08460), and [ExpertQA](https://arxiv.org/abs/2309.07852).
- We appreciate [🤗alignment-handbook](https://github.com/huggingface/alignment-handbook) for providing easy LM training framework!

## Citation
```bibtex
@misc{kim2025learningexploreselectcoverageconditioned,
      title={Learning to Explore and Select for Coverage-Conditioned Retrieval-Augmented Generation}, 
      author={Takyoung Kim and Kyungjae Lee and Young Rok Jang and Ji Yong Cho and Gangwoo Kim and Minseok Cho and Moontae Lee},
      year={2025},
      eprint={2407.01158},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01158}, 
}
```
