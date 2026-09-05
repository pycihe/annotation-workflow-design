# Annotations for Twitter Customer Care Dataset

This directory provides annotations for the Twitter Conversations Dataset for Conversational Document Prediction available on:
https://github.com/IBM/twitter-customer-care-document-prediction?tab=readme-ov-file

The annotations are separated into 3 text files which correspond to the dev, train and test files in the original dataset.

Each annotation file contains records with the following fields, separated by tabs:

- tweet_ID (from original dataset)
- customer tweet sentiment (annotation values: -2,-1,0,1,2)
- agent response intensity (annotation values: MISSING, WEAK, STRONG)
- Extracted/Rewritten header of agent response

The original customer tweets and agent responses can be retrieved from the original dataset using tweet_ID as the key. 

# Reference to the original dataset
```
@inproceedings{ganhotra-etal-2020-conversational,
    title = "Conversational Document Prediction to Assist Customer Care Agents",
    author = "Ganhotra, Jatin  and
      Roitman, Haggai  and
      Cohen, Doron  and
      Mills, Nathaniel  and
      Gunasekara, Chulaka  and
      Mass, Yosi  and
      Joshi, Sachindra  and
      Lastras, Luis  and
      Konopnicki, David",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.25",
    pages = "349--356",
}
```
