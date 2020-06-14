---
layout: post
title:  Locality Sensitive Hashing (LSH)
categories: [Data Mining, Tutorial]
excerpt:
mathjax: true
published: false
---

In the previous post we covered a method that approximates the Jaccard similarity by constructing a signature of the original representation. This allows us to significantly speed up the process of computing similarities between two sets. However, we still require to compare all pair of sets which can be extremely slow if the dataset is large. 
