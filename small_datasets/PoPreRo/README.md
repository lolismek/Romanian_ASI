# PoPreRo: A New Dataset for Popularity Prediction of Romanian Reddit Posts (ICPR 2024)

We introduce PoPreRo, the first dataset for Popularity Prediction of Romanian posts collected from Reddit. The PoPreRo dataset includes a varied compilation of post samples from five distinct subreddits of Romania, totaling 28,107 data samples. 

Along with our novel dataset, we introduce a set of competitive models to be used as baselines for future research. Interestingly, the top-scoring model achieves an accuracy of 61.35% and a macro F1 score of 60.60% on the test set, indicating that the popularity prediction task on PoPreRo is very challenging. Further investigations with based on few-shot prompting the Falcon-7B Large Language Model also point in the same direction. We thus believe that PoPreRo is a valuable resource that can be used to evaluate models on predicting the popularity of social media posts in Romanian.

# License
The dataset and code are released under: [Creative Commons Attribution Non Commercial Share Alike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

# 📑 Table of Contents
<a name = "tabel_of_contents"></a>

  - [Citation ](#citation-)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
  - [Usage ](#usage-)
  - [Developed with ](#️-developed-with-)

## Citation <a name = "citation"></a>
Please cite our work if you use any material released in this repository.
```
@inproceedings{Rogoz-ICPR-2024,
  author    = {Rogoz, Ana-Cristina and Nechita, Maria Ilinca and Ionescu, Radu Tudor},
  title     = "{PoPreRo: A New Dataset for Popularity Prediction of Romanian Reddit Posts}",
  booktitle = {Proceedings of the International Conference on Pattern Recognition},
  year      = {2024},
  }
```

## About <a name = "about"></a>

The PoPreRo dataset introduced in the paper "PoPreRo: A New Dataset for Popularity Prediction of Romanian Reddit Posts (ICPR 2024)" is available, together with the four baselines.

### Project structure

\ Dataset -  folder where the subreddit data is available. 


The proposed split of the five subreddits data was done as followed:

|            | Unpopular |     Unpopular      |  Popular       | Popular |    Total    |Total   | 
|------------|-----------|-----------|---------|---------|--------|---------|
| Set        | #posts    | #tokens   | #posts  | #tokens | #posts | #tokens |
| Train (Romanian subreddit)     | 12,053    | 398,219   | 11,592  | 560,580 | 23,645 | 958,799 |
| Validation (Bucuresti subreddit)| 1,059     | 75,742    | 1,054   | 80,297  | 2,113  | 156,039 |
| Test (Iasi, Timisoara, Cluj subreddit)      | 1,177     | 72,819    | 1,172   | 93,268  | 2,349  | 168,867 |

\ Mehods - folder which contains the four baselines as ipynb notebooks.



## Getting Started <a name = "getting_started"></a>

There are no environment dependencies that should be installed, although the dataset should be downloaded before running each of the the notebooks.

## Usage <a name="usage"></a>

The training and testing of the models is performed by running the notebook with the respective model name.

The available model options are:

- [Ro-BERT](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)
- [Ro-GPT2](https://huggingface.co/readerbench/RoGPT2-base)
- SVM
- Random Forest


## ⛏️ Developed with <a name = "developed_with"></a>

- [Pytorch](https://pytorch.org/) - Deep Learning Library.
- [PytorchLightning](https://www.pytorchlightning.ai/index.html) - Pytorch Framework
- [HuggingfaceTransformers](https://huggingface.co/)- Model Repository 


