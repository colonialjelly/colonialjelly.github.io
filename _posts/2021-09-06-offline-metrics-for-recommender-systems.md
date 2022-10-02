---
layout: post
title:  Offline Metrics for Recommender Systems
categories: [Tutorial]
excerpt:
mathjax: true
published: true
---
<!-- Evaluation for recommendation systems usually goes through two stages. The first stage of evaluation is referred to as offline evaluation. The goal is to select the best candidate system out of some number of candidate solutions. The measurement is usually done on some held-out historical data. As a result the best performing system on the offline metrics is picked to be deployed live. This usually entails bringing the system to some subset of the traffic and measuring some predefined KPI against an existing live system. -->

Evaluating recommender systems is notoriously tricky as offline measurements don't always align with online outcomes, but offline metrics nonetheless have an important place in the toolset of a recommender system's engineer. In this post, I'll cover some popular offline metrics that are used for evaluating recommender systems.

<!-- more -->

<!-- For all of the metrics, I'll assume that we have a recommender system that produces $$k$$ recommendations in descending order of relevance for each given query user, where $$k$$ is the maximum number of recommendations we're allowed to present to a user. -->

Since the whole point of a recommender system is to aid the user in discovery by reducing the amount of items they have to consider, we will assume that our recommender system is only allowed to make a maximum of $$k$$ recommendations for each user. We'll further assume that the recommendations are outputted as a ranked list, where higher positions imply that the recommender system has higher confidence or score for those items.

To give more concrete context to the metric calculation, let's imagine that we created a music recommendation system and we want to evaluate how well it works on some held-out data. Our evaluation data will be split across each user, that is for each user we'll have some data that will be fed to the recommender system as input and some hidden data that will be used to evaluate the output of the recommender system.  

<!-- The ground truth data is in the form of a list, with each item being a song that user has listened to and optionally we can also include how many times they've listened to it as well. -->

For each of the defined metric, I'll provide a simple Python implementation to make it easy to play around with different values and gain more intuition. These implementations are by no means efficient, they are simply meant to provide more insight.

#### Precision and Recall

Precision and recall are the most popular and probably the most intuitive metrics you can calculate. Recall measures what percentage of the user's liked items did we recommend, while precision computes what percentage of the recommended items were part of the user's liked items.

$$\text{Precision}_{k} = \frac{|\{\text{Liked Items}\}| \cap |\{\text{Recommended Items}\}|}{k}$$

$$\text{Recall}_{k} = \frac{|\{\text{Liked Items}\}| \cap |\{\text{Recommended Items}\}|}{|\{\text{Liked Items}\}|}$$

```python
def precision_k(relevant: list,
                predicted: list,
                k: int) -> float:
    return len(set(relevant).intersection(predicted[:k])) / k


def recall_k(relevant: list,
             predicted: list,
             k: int) -> float:
    return len(set(relevant).intersection(predicted[:k])) / len(relevant)
```

Notice that both metrics have the same numerator, you can use that fact to compute the precision and recall in one function and share the computed numerator between the two. Here's a tiny implementation in Python:

```python
def precision_recall_k(relevant: list,
                       predicted: list,
                       k: int) -> Tuple[float, float]:
    num_hits = len(set(relevant).intersection(predicted[:k]))
    precision_at_k = num_hits / k
    recall_at_k = num_hits / len(relevant)
    return precision_at_k, recall_at_k
```

For both recall and precision the values are in between 0 and 1, where 1 is the best possible value. Although, note that if the user has not liked at least $$k$$ items, then even the perfect system will get precision that is less than 1. Same goes for recall, if the user has greater than $$k$$ liked items, the recall of the best possible system will be less than 1.

#### F1-score

Precision and recall can be seen as a trade-off, we can usually arbitrarily increase recall by increasing $$k$$, the number of recommended items, however with higher $$k$$ the precision usually decreases. On the flip side, reducing $$k$$ usually leads to higher precision at the cost of lower recall. Having a metric that can capture both at the same time for a given, fixed $$k$$ would be great. That's exactly what F1-score does. It's the harmonic mean of precision and recall.

$$\text{F1}_{k} = \frac{2 \cdot \text{Precision}_{k} \cdot \text{Recall}_{k}}{\text{Precision}_{k} + \text{Recall}_{k}}$$

The F1-score is high when both recall and precision are high and is low when either one or both of them are low.  

```python
def f1_score(relevant: list,
             predicted: list,
             k: int) -> float:
    precision_at_k, recall_at_k = precision_recall_k(relevant,
                                                     predicted,
                                                     k)

    return (2 * precision_at_k * recall_at_k) / \
           (precision_at_k + recall_at_k)
```

#### Average Precision (AP) and Mean Average Precision (MAP)

One of the downsides of using precision and recall as a metric is the fact that they ignore the order of the recommendations. For instance, let's imagine we have two different recommender systems with the following outputs for some user:

$$\text{System}_A(\text{user}) = [6, 2, 1, 0, 3]$$

$$\text{System}_B(\text{user}) = [4, 1, 7, 2, 6]$$

And let's say that the only relevant items for the user are items 2 and 6. The precision and recall for both of the systems are identical, but we'd probably want System A to be preferred since it ranked the relevant items higher than System B. One of the ways this difference in the two systems can become apparent is if we plot a precision-recall (PR) curve by calculating precision and recall from 1 up to k for both systems.

<!-- [Insert PR curve] -->

But although plots are great for analysis, ideally we want the difference to be comparable using a single number. Luckily that's exactly what AP does. AP is (roughly) an approximation of the average area under the PR curve [^1]. To compute AP, we compute precision at each position that had a relevant recommendation for a user and then take the average of that.

<!--
For example, if the items at positions $$\{1, 4, 5\}$$ where relevant then we compute $$\text{Precision}_{1}, \text{Precision}_{4}$$ and $$\text{Precision}_{5}$$ and take the average. -->

<!-- AP gives more weight to correct predictions with a higher rank position in the recommendation, i.e. recommending a relevant item at position one will be more valuable than recommending it at position two. This means that we not only care about making relevant recommendations but we also care about how they are ordered. -->

<!-- To account for the order, AP measures precision at positions that had a relevant recommendation. For example, if the items at positions $$\{1, 4, 5\}$$ where relevant then we compute $$\text{Precision}_{1}, \text{Precision}_{4}$$ and $$\text{Precision}_{5}$$ and just take the average. -->

$$\text{AP}_{k} = \frac{1}{n} \sum_{i=1}^{k} \text{Precision}_{i} * \text{rel}_{i}$$

Where $$\text{rel}_{i}$$ is equal to 1 if item $$i$$ is relevant and 0 otherwise and $$n$$ is the number of relevant items in the entire recommendation set.

```python
def ap_k(relevant: list,
         predicted: list,
         k: int) -> float:
    relevant_idx = []
    # Find indices of predicted items that were relevant
    for i, item in enumerate(predicted[:k]):
        if item in relevant:
            relevant_idx.append(i + 1)

    # Compute precision at each index of predicted relevant item
    precisions = []
    for idx in relevant_idx:
        # Using the precision_k function we defined earlier
        precision_at_idx = precision_k(relevant, predicted, idx)
        precisions.append(precision_at_idx)

    return float(np.mean(precisions))
```

Now, the MAP is just the mean of APs across a collection of users.

$$\text{MAP}_{k} = \frac{1}{N} \sum_{j=1}^{N} \text{AP}_{k}(j)$$

```python
def map_k(relevant_batch: List[list],
          predicted_batch: List[list],
          k: int) -> float:
    return np.mean([ap_k(relevant, predicted, k)
                    for relevant, predicted
                    in zip(relevant_batch, predicted_batch)])
```

#### Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)

Different from the previous metrics, reciprocal rank only cares about the rank of the first relevant recommendation. Let the $$\text{rank}_{k}(\text{user}_{i})$$ be a function that returns the rank of the first relevant item in the $$k$$ ranked recommendations for user $$i$$. The reciprocal rank (RR) is then defined as:

$$\text{RR}_k = \frac{1}{\text{rank}_{k}(\text{user}_{i})}$$

Note that RR is undefined if the $$k$$ recommendations do not contain any relevant items, in such a case we set the RR to zero.

```python
def rr_k(relevant: list,
         predicted: list,
         k: int) -> float:
    rank = 0
    for i, item in enumerate(predicted[:k]):
        if item in relevant:
            rank = i + 1
            break
    return 1. / rank if rank else 0.
```

Now, MRR is just the mean of RRs over a collection of users $$U$$:

$$\text{MRR}_k = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{\text{rank}_{k}(\text{user}_{i})} = \frac{1}{N}\sum_{i=1}^{N} \text{RR}_{i}$$

Where $$N$$ is the number of users $$U$$. We can utilize the implementation for RR to implement the MRR:

```python
def mrr_k(relevant_batch: List[list],
          predicted_batch: List[list],
          k: int) -> float:
    return np.mean([rr_k(relevant, predicted, k)
                    for relevant, predicted
                    in zip(relevant_batch, predicted_batch)])
```

#### Normalized Discount Cumulative Gain (nDCG)

<!-- So far, all of the metrics we discussed assume that item relevancy is binary, i.e. either something is relevant to the user or not. But often times in real applications, we not only know wether someone likes something or not but we also have some information on how much someone likes a particular item. For example, for our music recommender maybe instead of just knowing what songs the user liked we also know how much they liked it by using the listen counts on those songs.

e.g. we can take how many times they listened to a song and use that to measure how much they like the song. This is where nDCG comes in, it's a special kind of metric that not only takes the order into account but also the relevancy scores of the predicted items. -->

So far, all of the metrics we discussed assume that item relevancy is binary, i.e. either something is relevant to the user or not. But often times in real applications, we not only know if an item is relevant, but we also have some information on *how* relevant the item is to the user. Going back to the music recommender, instead of just looking at what songs the user liked, we could additionally consider the listen counts and use it to measure the degree of relevancy. In this setting, the goal will be to recommend items that have high degree of relevancy to the user at higher positions in the recommendation output. In order to understand if we're achieving that, we need a metric like nDCG.


<!-- we not only know wether someone likes something or not but we also have some information on how much someone likes a particular item. For example, for our music recommender, we could measure how much a user likes a song by counting the number of listens. We can say that the more times a user has listened to a song the higher that user rates that song. In order to take this information into account, we need a metric that uses non-binary relevancy scores. This is where nDCG comes in, it's a scary sounding metric but it's actually quite simple. The first letter just stands for normalized, which we'll get to later. Let's first focus on DCG.


But what if we have non-binary relevancy. For example, for our music recommender maybe instead of just knowing what songs the user liked we also know how much they liked it, e.g. we can take how many times they listened to a song and use that to measure how much they like the song. This is where nDCG comes in, it's a special kind of metric that not only takes the order into account but also the relevancy scores of the predicted items. -->

<!--
it's a scary sounding metric but it's actually quite simple. The first letter just stands for normalized, which we'll get to later. Let's first focus on DCG. -->

Let $$s_{i}$$ be the relevancy score for the item at position $$i$$ in our recommended item list. Then the DCG is computed as:

$$DCG_{k} = \sum_{i=1}^{k}\frac{s_{i}}{\log_{2}({i + 1})}$$

We normalize the DCG by dividing it by the best possible DCG score that is achievable for the given user. We refer to this quantity as the ideal discount cumulative gain or IDCG for short. The IDCG is simply the score we would get if we recommended all the user's relevant items in descending order of relevancy score. The nDCG then is computed as the ratio of the DCG and the IDCG:

<!-- We normalize the DCG to make it between 0 and 1, we compute the best possible DCG score that is achievable on the given list of items. Which would be achieved by an ordering that has the highest scoring item by relevancy as the first item, the second highest as the second item and so on. We refer to the ideal score as $$IDCG$$. The nDCG then is computed as the ratio of the DCG and the IDCG: -->

<!-- To normalize the DCG, we compute the best possible DCG score that is achievable on the given list of items. Which would be achieved by an ordering that has the highest scoring item by relevancy as the first item, the second highest as the second item and so on. We refer to the ideal score as $$IDCG$$. The nDCG then is computed as the ratio of the DCG and the IDCG: -->

$$nDCG_{k} = \frac{DCG_{k}}{IDCG_{k}}$$

```python
import numpy as np

def ndcg_k(relevant: list,
           relevancy_scores: list,
           predicted: list,
           k: int = None) -> float:
    # Create a relevancy array for the predicted items
    relevancy_scores_dict = dict(zip(relevant, relevancy_scores))
    predicted_item_relevancy_scores = []
    for item in predicted:
        if item in relevant:
            predicted_item_relevancy_scores.append(relevancy_scores_dict[item])
        else:
            predicted_item_relevancy_scores.append(0)

    # Convert it to a ndarray
    predicted_item_relevancy_scores = np.array(predicted_item_relevancy_scores)

    # Compute the DCG
    dcg = _dcg_k(predicted_item_relevancy_scores, k)

    # Compute the ideal DCG
    idcg = _dcg_k(np.sort(relevancy_scores)[::-1], k)

    # return normalized DCG
    return dcg / idcg


def _dcg_k(relevancy_scores: list, k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(1, min(k, len(relevancy_scores)) + 1) + 1)
    return float(np.sum(relevancy_scores[:k] * discounts))
```

<!-- #### Coverage

One of the simplest metrics one can look at that does not require any held-out data for users is the item coverage. Item coverage is the percentage of items from the entire item space is provided as a recommendation to at least one user.

$$\text{Coverage} = \frac{\text{Number of recommended items}}{\text{Number of total items}}$$

This metric essentially measures how much of the item space the recommendation system will allow the users to explore through the provided recommendations. -->


[^1]: For more information, check out [ Evaluation of ranked retrieval results ](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
