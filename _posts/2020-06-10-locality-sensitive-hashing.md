---
layout: post
title:  Locality Sensitive Hashing
categories: [Data Mining, Tutorial]
excerpt:
mathjax: true
published: false
---

In the previous post we covered a method that approximates the Jaccard similarity by constructing a signature of the original representation. This allows us to significantly speed up the process of computing similarities between the sets. But remember that the goal is to find all similar items to a given item. This requires to compute the similarities between all pairs of items in the data. If we go back to our example, Spotify has about 1.2 million artists on their platform. Which means that to find all similar artists we need to make 1.4 trillion comparisons... ahm... how about no. We're going to do something different. We're instead going to use Locality Sensitive Hashing (LSH) to identify candidate pairs and only compute the similarities on those. This will substantially reduce the computational burden.

LSH is a neat method to find similar items without computing similarities between every pair. It accomplishes this by hashing items in such a way that results in similar items being hashed in the same bucket with high probability. If two items are hashed to the same bucket, we consider them as candidate pairs and proceed with computing their similarity.

### MinHash LSH

The method that we're going to describe relies on having a signature matrix of all of our sets precomputed. It is a pretty easy procedure both algorithmically and conceptually. It uses the intuition that if two items have identical parts in some random positions then they're probably similar. This is the idea we're going to turn to, in order to identify candidate pairs.

We begin by dividing the signature matrix into $$b$$ bands with $$r$$ rows. We hash each item's portion of the band with some hashing function [^1]. We use the same hash function for every band but each band should have a separate array to store the buckets. If for a given band two items end up hashed into the same bucket, we consider them as candidate pairs. What this means is that if two items match each other on some portion of the signature we count them as candidates. Using a hashing function rather than directly comparing them is what allows us to avoid the quadratic amount of comparisons. Here's a really simple implementation of an LSH for Jaccard similarities:

```python
from collections import defaultdict
import numpy as np

def minhash_lsh(sig, b):
    num_rows = sig.shape[1]
    bands = np.split(sig, b)
    bands_buckets = []
    for band in bands:
        items_buckets = defaultdict(list)
        items = np.hsplit(band, num_rows)
        for i, item in enumerate(items):
            item = tuple(item.flatten().astype(int))
            items_buckets[item].append(i)
        bands_buckets.append(items_buckets)

    return bands_buckets
```



[^1]: We can use the built-in hashing function of whatever programming language we're using.
