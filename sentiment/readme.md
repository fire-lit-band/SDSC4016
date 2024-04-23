Here are the detailed explanations for the code in this folder

# doing token

Since the tokenization process takes a lot of time, we decide to store the results of tokenization as "pkl" to reduce the tokening time

# no-pretrained-bert

It is the main code to do the sentiment analysis. Here the "no-pretrained" means that we directly use the "bert-base-uncased" model written from Google.

## pretrained-bert

"pretrained" here means that we use the pretrained model from Yelp 2015 polarity dataset done by this paper:

[[1905.05583\] How to Fine-Tune BERT for Text Classification? (arxiv.org)](https://arxiv.org/abs/1905.05583)

Since the results are almost the same as "no-pretrained" one. So we do not show it in our present and report.

# senti_results

The code is to generate the results of sentiment analysis and to compute its accuracy

# enhance_senti

This method is used to use the results from sentiment analysis to enhance the original rating matrix

