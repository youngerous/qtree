<div align="center">
    
# Learning to Explore and Select for Coverage-Conditioned Retrieval-Augmented Generation

[![Paper](https://img.shields.io/badge/Paper-arxiv.2407.01158-green)](https://arxiv.org/abs/2407.01158)

#### Takyoung Kim<sup>1,&ast;</sup>, Kyungjae Lee<sup>2</sup>, Young Rok Jang<sup>2</sup>, Ji Yong Cho<sup>2,3</sup>, Gangwoo Kim<sup>4,&ast;</sup>, Minseok Cho<sup>2</sup>, Moontae Lee<sup>2,5</sup> <br> <sub><sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>LG AI Research, <sup>3</sup>Cornell University, <sup>4</sup>Korea University, <sup>5</sup>University of Illinois Chicago</sub> <br> <sub><sup>&ast;</sup>Work done as a research intern at LG AI Research</sub>

</div>

## Abstract

> Interactions with billion-scale large language models typically yield long-form responses due to their extensive parametric capacities, along with retrieval-augmented features. While detailed responses provide insightful viewpoint of a specific subject, they frequently generate redundant and less engaging content that does not meet user interests. In this work, we focus on the role of query outlining (i.e., selected sequence of queries) in scenarios that users request a specific range of information, namely coverage-conditioned ($C^2$) scenarios. For simulating $C^2$ scenarios, we construct QTree, 10K sets of information-seeking queries decomposed with various perspectives on certain topics. By utilizing QTree, we train QPlanner, a 7B language model generating customized query outlines that follow coverage-conditioned queries. We analyze the effectiveness of generated outlines through automatic and human evaluation, targeting on retrieval-augmented generation (RAG). Moreover, the experimental results demonstrate that QPlanner with alignment training can further provide outlines satisfying diverse user interests.

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
@misc{kim2024learning,
      title={Learning to Explore and Select for Coverage-Conditioned Retrieval-Augmented Generation}, 
      author={Takyoung Kim and Kyungjae Lee and Young Rok Jang and Ji Yong Cho and Gangwoo Kim and Minseok Cho and Moontae Lee},
      year={2024},
      eprint={2407.01158},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01158}, 
}
```
