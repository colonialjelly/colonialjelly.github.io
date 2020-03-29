---
layout: post
title:  Min Hashing
categories: [Data Mining, Educational]
excerpt:
mathjax: true
published: true
---


Suppose you're an engineer at Spotify and you're on a mission to create a feature that lets users explore new artists that are similar to the ones they already listen to. First thing you need to do is represent the artists in such a way that they can be compared to each other. You figure that one obvious way to characterize an artist is by the people that listen to it. You decide that each artist shall be defined as a set of user ids of people that have listened to that artist at least once. For example, the representation for Miles Davis could be,

$$\text{Miles Davis} = \{5, 23533, 2034, 932, ..., 10003243\}$$

The number of elements in the set is the number of users that have listened to Miles Davis at least once. To compute the similarity between artists, we can compare these set representations (more on this later). Now, with Spotify having more than 271 million users, these sets could be very large (especially for popular artists). It would take forever to compute the similarities, especially since we have to compare every artist to each other. In this post, we're going to talk about a method that can help us speed up this process. We're going to be converting each set into a smaller representation called a signature, such that the similarities between the sets are well preserved.

<!-- We're going to do something different instead. Instead of representing the artist as a set of all of the users, that listen to it.

In this post, we're going to talk about how to speed up the process of computing the similarities between these sets. -->

<!-- That is, you compute the ratio between the size of the intersection of the sets and the union. This -->

<!-- Suppose we want to cluster similar artists on Spotify. With about 271 million artists, if we assume that each artist has about 1000 songs in their listening history we would need ~4.33 terabytes to represent the entire data. That is a lot! To make things even worse, in order to find the clusters with $$n$$ artists, we need to measure the similarity between all $${n \choose 2} = \frac{n(n - 1)}{2}$$ pairs of artists. With 271 million artists that is an astronomical number of comparisons!

In this post, the goal will be to reduce the size of the representation of each artist while preserving the similarities. This will take care of the memory issue and some of the computational burden. In the next post, we'll talk about what to do with the bigger computational problem of calculating similarities between all of the pairs of artists. -->

<!--
Measuring similarity of objects is one of the most fundamental computations for data mining. Similarity can be used to detect plagiarism, categorize documents, recommend products to customers and there are many many more applications. There are a lot of different ways of defining similarity. In this post I'll be talking about Jaccard similarity and its' approximation. -->

### Toy example

I think working with tiny examples to build intuition is an excellent method for learning. So in that spirit, let's consider a toy example. Let's assume that we only have 3 artists and we have a total of 8 users in our dataset.

$$\text{artist}_{1} = \{1, 4, 7\}$$

$$\text{artist}_{2} = \{0, 1, 2, 4, 5, 7\}$$

$$\text{artist}_{3} = \{0, 2, 3, 5, 6\}$$

<!-- I think working with tiny examples to build intuition is an excellent method for learning. So in that spirit, let's consider a toy example. Let's assume that we only have 3 artists and we have a total of 8 songs in our dataset. -->

<!-- ![](../images/artist_data.png) -->

### Jaccard similarity

The goal is to find similar artists, so we obviously need a way to measure similarities between a pair of artists. We will be using Jaccard similarity. It is defined as the fraction of shared elements between two sets. In our case the sets are user ids. All we have to compute is how many users each pair of artists share with each other, divided by total number of users in both artists. For example the Jaccard similarity between $$\text{artist}_1$$ and $$\text{artist}_2$$

<!-- To compute the similarities, we said that we were going to measure the fraction of shared users for each artist. This computation actually has a name, it's called the Jaccard similarity. For a pair of artists, we compute the Jaccard similarity by counting the number of users that are shared and dividing it by the total number of users in both of the artist sets. For example the Jaccard similarity between $$\text{artist}_{1}$$ and $$\text{artist}_{2}$$ -->

<!-- [^renamed_artists]: To make things a little easier to write, I'll refer to the artists by their subscripts from now on. For example. $$\text{artist}_{A} \rightarrow A$$ and so on. -->

<!-- The goal is to cluster similar artists, so we obviously need a way to measure similarities between a pair of artists. We will be using Jaccard similarity. It is defined as the fraction of shared elements between two sets. In our case the sets are artists' listening history. All we have to compute is how many songs each pair of artists share with each other, divided by total number of songs in both artists. For example the Jaccard similarity between $$\text{artist}_1$$ and $$\text{artist}_2$$ -->

$$J(\text{artist}_{1}, \text{artist}_{2}) = \frac{|\text{artist}_{1} \cap \text{artist}_{2}|}{|\text{artist}_{1} \cup \text{artist}_{2}|} = \frac{|\{1, 4, 7\}|}{|\{0, 1, 2, 4, 5, 7\}|} = \frac{3}{6} = 0.5$$

Similarly for the other pairs, we have:

<center>
$$J(\text{artist}_{2}, \text{artist}_{3}) = \frac{3}{8} = 0.375$$
$$J(\text{artist}_{1}, \text{artist}_{3}) = \frac{0}{8} = 0$$
</center>

A few important facts about the Jaccard similarity:

 - The Jaccard similarity is 0 if the two sets share no elements, and it's 1 if the two sets are identical. Every other case has values between 0 and 1.
 - The Jaccard similarity between two sets corresponds to the probability of a randomly selected element from the union of the sets also being in the intersection.

 Let's unpack the second one, because it's definitely the most important thing to know about the Jaccard similarity.

<!-- An important fact about the Jaccard similarity is that it corresponds to the probability that a randomly selected element in the union is also in the intersection. This is a crucial property of the Jaccard similarity that is central to understanding why min hashing works. -->

### Intuition behind the Jaccard similarity

For some people (present company included), visual explanations are easier to grasp than algebraic ones. We'll briefly shift our view from sets to Venn diagrams. Let's imagine any two artists as Venn diagrams, the Jaccard similarity is the size of the intersection divided by the size of the union:

<img src="../images/js_venn.png" width="50%">

Now imagine that I'm throwing darts on the diagrams and I'm horrible at it. I'm so bad that every element on the diagrams has an equal chance of being hit. What's the chance that I throw a dart and it lands on the intersection? It would be the number of elements in the intersection divided by the total number of elements. Which is exactly what the Jaccard similarity is. This implies that the larger the similarity, the higher the probability that we land on the intersection with a random throw.

<img src="../images/venns.png" width="50%">

Consider another scenario. Suppose you want to know the similarity between two sets, but you can't see the diagram, you're blindfolded. However, if you throw a dart, you do get the information on where it landed. Can you make a good guess on the similarity of two sets by randomly throwing darts on it? Let's say after throwing 10 darts we know that 6 of them landed in the intersection. What would you guess that the similarity of the two sets are? Let's say after throwing 40 more darts, we know that 30 of the total 50 throws landed in the intersection. What would your guess be now? Are you more certain about your guess? Why?

Ponder about this for a little bit and keep this picture in mind throughout reading this post. This is, in essence, the basis for the MinHash algorithm.


### Approximate Jaccard similarity

In the previous paragraph, we have alluded to the fact that it's possible to approximate the Jaccard similarity between two sets. In order to see why that's true, we need to rehash some of things we've said, mathematically.

Let's take $$\text{artist}_1$$ and $$\text{artist}_2$$ and their union $$\text{artist}_1 \cup \text{artist}_2 = \{0, 1, 2, 4, 5, 7\}$$.
Some of the elements in union are also in the intersection, more specifically $$\{1, 4, 7\}$$.

Let's replace the elements with the symbols "-" and "+", denoting if an element appears in the intersection or not.

$$\{0, 1, 2, 4, 5, 7\} \rightarrow \{-, +, -, +, -, +\}$$

If every element has an equal probability of being picked, what is the probability of drawing an element that is of type "+"? It's the number of pluses divided by number of pluses and number of minuses.

$$P(\text{picking a "+"}) = \frac{\text{number of "+"}}{\text{number of "+" and "-"}}$$

The number of "+" corresponds to the number of elements in the intersection and the number of "+" and the number of "-" corresponds to the total number of elements or the size of the union. Therefore,

$$P(\text{picking a "+"}) = \frac{\text{number of "+"}}{\text{number of "+" and "-"}} = \frac{|\{1, 4, 7\}|}{|\{0, 1, 2, 4, 5, 7\}|} = J(\text{artist}_1, \text{artist}_2)$$

<!-- More than two sets:

If we have more than two sets some things change. Now we can have multiple intersections. Just knowing that a dart landed in an intersection is not enough, we need to keep track of all possible intersections between the sets. This might be too much of a hassle. So we're gonna change the game.

Imagine that you're at a carnival and there's a shooting game. There are n diagrams on the board, you are again blindfolded. The game is to shoot for k rounds and guess the similarities between all the sets with **some** tolerance for error. Each round consists of up to n throws of a dart. Let's imagine that each throw has a guarantee to hit at least one diagram but it could potentially hit more than one (if it lands in an intersection). After a diagram gets hit, it gets eliminated. The round is over when all of the diagrams are eliminated. The diagrams get reset after each round. As before, you get to know where the dart lands, that is, you get to know the exact element that you hit and which of the diagrams were eliminated. Can you come up with a way to guess the similarities between all of the pairs of sets? -->

<!-- Let $$X$$ be a random variable such that $$X=1$$ if we draw a plus and $$X=0$$ if we draw a minus.

$$\mathbb{E}[X] = P(X=1) \times 1 + P(X=0) \times 0 = 0.5 = J(\text{artist}_{1}, \text{artist}_{2})$$ -->

What this means is that we can approximate the Jaccard similarity between pairs of artists. Let $$X$$ be a random variable such that $$X = 1$$ if we draw a plus and $$X = 0$$ if we draw a minus. $$X$$ is a Bernoulli random variable with $$p=J(\text{artist}_{1}, \text{artist}_{2})$$. In order to estimate the similarity we can estimate $$p$$. In this case we obviously know that $$p=0.5$$ since we already computed it, but let's assume that we don't know this.

If we repeat the random draw multiple times and keep track of how many times a "+" type came up versus a "-", we can estimate the parameter $$p$$ for $$X$$ by maximum likelihood estimation (MLE):

$$\hat{p}  = \frac{1}{n} \sum_{i=1}^{n} X_{i} = \hat{J}(\text{artist}_{1}, \text{artist}_{2})$$

Where $$X_{i}$$ are our observations and $$n$$ is the total number of draws that were made. The larger the number of draws $$n$$, the better the estimation will be.

The code below will simulate the process a 30 times and empirically compute the similarity.

```python
import numpy as np

num_trials = 30

# Union of artist_1 and artist_2
union = np.array([0, 1, 2, 4, 5, 7])

# Intersection of artist_1 and artist_2
intersection = np.array([1, 4, 7])

# Randomly pick element
draws = np.random.choice(union, num_trials)
num_intersect = np.count_nonzero(np.isin(draws, intersection))
print(num_intersect/len(draws))

```

If you run the above code you should get something that is close to $$0.5$$. Which, as expected, corresponds to the Jaccard similarity between $$\text{artist}_1$$ and $$\text{artist}_2$$. Play around with the variable $$\text{num_trials}$$, what happens if you set it to 1? What about 10,000?

<!-- So what does this mean? This means that we can approximate Jaccard similarity using randomness. We're going to be using this fact to come up with a way to encode the original data into a smaller representation called a *signature* such that the Jaccard similarities are well approximated.  -->


<!-- This means that this random process in expectation is the same as the Jaccard similarity. We're going to be using this fact to come up with a way to encode the original data into a smaller representation such that the Jaccard similarities are well approximated. -->

<!-- Just to restate the goal, we have a dataset $$D$$ that we want to encode in some smaller dataset $$D^{'}$$ such that $$J_{pairwise}(D) \approx J_{pairwise}(D^{'})$$. Where $$J_{pairwise}$$ is the pairwise Jaccard similarity of all artists in the data. -->

<!-- This is great, but we need to compute similarities between all pairs of artists not just two artists. -->

### Shuffling and picking first $$\equiv$$ Randomly picking

Before we move on, we need to understand one more thing. Randomly selecting an element from a set is the same thing as shuffling the set and picking the first element [^1]. Everything that we have said above is also true if we, instead of randomly selecting an element, shuffled the set and picked the first element. Make sure to pause here, if this doesn't make sense.

[^1]: I know shuffling a set of elements is meaningless since sets don't have order but imagine that they do :).

```python
import numpy as np

num_trials = 30

# Union of artist_1 and artist_2
union = np.array([0, 1, 2, 4, 5, 7])

# Intersection of artist_1 and artist_2
intersection = np.array([1, 4, 7])

# Shuffle and pick first element
num_intersect = 0
for i in range(num_trials):
    np.random.shuffle(union)
    if union[0] in intersection:
        num_intersect += 1
print(num_intersect/num_trials)
```
The code above implements the same process that I described before, but instead of randomly picking an element it is shuffling the elements in the union and picking the first element. If you run this you should similarly get something that is close to $$0.5$$.

### Data matrix

We have shown that it's possible to approximate Jaccard similarity for a pair of artists using randomness. But our previous method had a significant issue. We still needed to have the intersection and the union set to estimate the Jaccard similarity, which kind of defeats the whole purpose. We need a way to approximate the similarities without having to compute these sets. We also need to approximate the similarities for all pairs of artists, not just a given pair. In order to do that, we're going to switch our view of the data from sets to a matrix.

<img src="../images/artist_matrix.png" width="50%">

The columns represent the artists and the rows represent the user IDs. A given artist has a $$1$$ in a particular row if the user with that ID has that artist in in their listening history [^2].

[^2]: Obviously, we wouldn't store the data in this form in practice, since it's extremely wasteful. But seeing the data as a matrix will be a helpful for conceptualizing the methods that we're gonna discuss.

### Min Hashing

Going back to our main goal. We want to reduce the size of the representation for each artist while preserving the Jaccard similarities between pairs of artists in the dataset. In more "mathy" terms, we have a data matrix $$D$$ that we want to encode in some smaller matrix $$\hat{D}$$ called the signature matrix, such that $$J_{pairwise}(D) \approx \hat{J}_{pairwise}(\hat{D})$$

[^4]: $$J_{pairwise}$$ is a function that produces a matrix which represents all pairwise similarities of the artists in the data. -->

<!-- To reiterate the goal, we want to encode the data into a smaller representation such that the Jaccard similarities are preserved. In more "mathy" terms, we have a data matrix $$D$$ that we want to encode in some smaller matrix $$\hat{D}$$ called the signature matrix, such that $$J_{pairwise}(D) \approx \hat{J}_{pairwise}(\hat{D})$$ [^4].

[^4]: $$J_{pairwise}$$ is a function that produces a matrix which represents all pairwise similarities of the artists in the data. -->

The first algorithm I will be describing is not really practical but it's a good way to introduce the actual algorithm called MinHash. The whole procedure can be summarized in a sentence: shuffle the rows of the data matrix and for each artist (column) store the ID of the first non-zero element. That's it!

**naive-minhashing**
```
for k iterations
  shuffle rows
  for each column
    store the ID of first non-zero element into the signature matrix
```
Let's go through one iteration of this algorithm:

![](../images/1iteration.png)

We have reduced each artist to a single number. To compute the Jaccard similarities between the artists we compare the signatures. Let $$h$$ be the function that finds and returns the index of the first non-zero element. Then we have:

<center>
$$h(\text{artist}_{1}) = 7$$
$$h(\text{artist}_{2}) = 0$$
$$h(\text{artist}_{3}) = 0$$
</center>

And the Jaccard similarities are estimated as  [^3]:

$$\hat{J}(\text{artist}_{i}, \text{artist}_{j}) = \unicode{x1D7D9}[h(\text{artist}_{i}) = h(\text{artist}_{j})]$$

[^3]: The strange looking one ($$\unicode{x1D7D9}$$) is called the indicator function. It outputs a 1 if the expression inside the brackets evaluates to true, otherwise the output is a 0.


**Why would this work?**

To understand why this is a reasonable thing to do, we need to recall our previous discussion on approximating the Jaccard similarity. We were drawing elements from the union of two sets at random and checking if that element appeared in the intersection. What we're doing here might look different, but it's actually the same thing.

- We are shuffling the rows (thus bringing in randomness)
- By picking the first non-zero element for every artist, we're always picking an element from the union (for any pair of artists).  
- By checking if $$h(\text{artist}_{i}) = h(\text{artist}_{j})$$ we are checking if the element is in the intersection
- And most importantly the probability of a randomly drawn element being in the intersection is exactly the Jaccard similarity, that is, $$P(h(\text{artist}_{i}) = h(\text{artist}_{j})) = J(\text{artist}_{i}, \text{artist}_{j})$$


<!-- for any two sets, we're always sampling from the union, and when do we have a success? When both of the rows have a one, i.e $$h(\text{artist}_{i}) = h(\text{artist}_{j})$$ i.e++ the element is in the intersection. Sound familiar?  

Let $$Y$$ be a random variable that is 1 if $$h(\text{artist}_{i}) = h(\text{artist}_{j})$$ and is 0 otherwise. Now what's $$p = P(h(\text{artist}_{i}) = h(\text{artist}_{j}))$$, it is none other than $$J(\text{artist}_{i}, \text{artist}_{j})$$, that is, we're claiming that

$$P(h(\text{artist}_{i}) = h(\text{artist}_{j})) = J(\text{artist}_{i}, \text{artist}_{j})$$ -->

<!-- $$Y$$ is a Bernouli random variable with parameter $$p = J(\text{artist}_{i}, \text{artist}_{j})$$. Hopefully, things should be coming back now. How can we estimate $$p$$, the same way as before. We do multiple trials and estimate $$p$$ as the average. -->

<!-- $$\hat{J}(\text{artist}_{i}, \text{artist}_{j}) = \frac{1}{k}\sum_{l=1}^{k} = \unicode{x1D7D9}[h_{l}(\text{artist}_{i}) = h_{l}(\text{artist}_{j})]$$ -->

<!-- The probability that $$h(\text{artist}_{i})$$ and $$h(\text{artist}_{j})$$ are the same is exactly $$J(\text{artist}_{i}, \text{artist}_{j})$$. That is, we're claiming that $$P(h(\text{artist}_{i}) = h(\text{artist}_{j})) = J(\text{artist}_{i}, \text{artist}_{j})$$. What this means is that, if a pair of artists have a high similarity, there is a high probability they will have the same value for $$h$$. Do you remember throwing darts at the diagrams? The intuition is the same here. -->

Let's go through an example together with sets $$\text{artist}_{1}$$ and $$\text{artist}_{2}$$. I've highlighted the relevant rows using the same definition for the symbols "+" and "-" as before. We have an additional symbol called "null", these correspond to elements that are in neither of the selected artists. The "null" type rows can be ignored since they do not contribute to the similarity (and they are skipped over in the algorithm).

<img src="../images/artist_matrix_highlighted.png" width="50%">

If we shuffled the rows what is the probability that the first **non**-"null" row is of type "+"? In other words, after shuffling the rows, if we proceeded from top to bottom while skipping over all "null" rows, what is the probability of seeing a "+" before seeing a "-"?

If we think back to our example with sets, this question should be easy to answer. All we have to realize is that,
encountering a "+" before a "-" is the exact same thing as randomly drawing a "+" in the union. Which we know has a probability that is equal to the Jaccard similarity.

<!-- [^3]: The only difference is that we're using shuffling instead of randomly picking, which we've concluded is the exact same thing. -->

$$P(\text{seeing a "+" before "-"})  = \frac{\text{number of "+"}}{\text{number of "+" and "-"}} = J(\text{artist}_{1}, \text{artist}_{2})$$

If the first row is of type "+" that also means that $$h(\text{artist}_{1}) = h(\text{artist}_{2})$$, so the above expression is equivalent to saying:

$$P(h(\text{artist}_{1}) = h(\text{artist}_{2})) = J(\text{artist}_{1}, \text{artist}_{2})$$

The same argument holds for any pair of artists. The most important take away here is that if the Jaccard similarity is high between two pairs of sets, then the probability that $$h(\text{artist}_{i}) = h(\text{artist}_{j})$$ is also high. Remember, throwing darts at the diagrams? It's the same intuition here.

Now going back to our example. With a single trial we have the following estimations.

<!-- So $$Y$$ is a Bernoulli random variable with parameter $$p = J(\text{artist}_{i}, \text{artist}_{j})$$. How can we estimate $$p$$, the same way we did before, by simulating multiple trials and taking the average. -->

<!-- Remember, throwing darts at the diagrams? That's exactly what we're doing here. You can think of this process as throwing a dart on the diagram and then checking if it landed in an intersection. -->

<!-- Going back to our example, we have pairs ($$\text{artist}_1$$, $$\text{artist}_2$$) and ($$\text{artist}_1$$, $$\text{artist}_3$$) having similarity zero since their signatures do not match. The similarity for ($$\text{artist}_2$$, $$\text{artist}_3$$) will be 1 since both have the same signature. -->


<img src="../images/sig1.png" width="50%">

As you can see it's a *little* off. How can we make it better? It's simple, we run more iterations and make the signatures larger. In the earlier discussions we introduced a Bernoulli random variable and we estimated it's parameter by simulating multiple random trials. We can do the same exact thing here. Let $$Y$$ be a random variable that has value 1 if $$h(\text{artist}_{i}) = h(\text{artist}_{j})$$ and is 0 otherwise. $$Y$$ is an instance of a Bernoulli random variable with $$p = J(\text{artist}_{i}, \text{artist}_{j})$$. If we run the algorithm multiple times, thus simulating multiple but identical variables $$Y_{1}, Y_{2}, ..., Y_{k}$$, we can then estimate the Jaccard similarity as:

<!-- Remember how we approximated the parameter $$p$$ for the random variable $$X$$? It's the exact same thing here. With a signature with length greater than one the estimation for Jaccard similarity is done by taking the average of each element-wise comparison. -->

$$\hat{J}(\text{artist}_{i}, \text{artist}_{j}) = \frac{1}{k}\sum_{m=1}^{k}Y_{m} = \frac{1}{k}\sum_{l=1}^{k} \unicode{x1D7D9}[h_{m}(\text{artist}_{i}) = h_{m}(\text{artist}_{j})]$$

Where $$h_{m}$$ is a function that returns the first non-zero index for iteration $$m$$.

<!-- In the earlier parts of the post, we defined a Bernoulli random variable $$X$$. Do you see similarities to that and what we're doing now? -->

<!-- This is because we're only using a single signature to measure the similarities. This corresponds to only having a single trial in the random experiments we defined previously. As before, the more trials we have, the better the estimation will be.

 We've mentioned before that the more random simulations we run the better the approximation will be. In order to have a better approximation, we should run a few more iterations of this process. This would result in a larger signature matrix. -->

The animation below shows the process of going through 3 iterations of this algorithm:

![](../images/minhashing_permuation_animation.gif)

Computing the Jaccard similarities with the larger signature matrix:

<img src="../images/sig3_sims.png">

That's much better. It's still not exactly the same but it's not too far off. We've managed to reduce the number of rows of the matrix from 8 to 3 while preserving the pairwise Jaccard similarities up to some error. To achieve a better accuracy, we could construct an even larger signature matrix, but obviously we would be trading off the size of the representation.

If you want to play around with this algorithm, here's an implementation in Python using Numpy:

```python
import numpy as np

def min_hashing_naive(data, num_iter):
    num_artists = data.shape[1]
    sig = np.zeros(shape=(num_iter, num_artists), dtype=int)
    for i in range(num_iter):
        np.random.shuffle(data)
        for j in range(num_artists):
            c = data[:, j]
            if np.any(c != 0):
                min_hash = np.where(c == 1)[0][0]
                sig[i, j] = min_hash
    return sig
```

### MinHash Algorithm

Shuffling the rows of the data matrix can be infeasible if the matrix is large. In the Spotify example we would have to shuffle 271 million rows for each iteration of the algorithm. So while the algorithm works conceptually, it is not that useful in practice.

Instead of explicitly shuffling the rows, what we can instead do is *implicitly* shuffle the rows by mapping each row index to some unique integer. There are special functions called hash functions that can do exactly that. They map each unique input to some unique output (usually in the same range).

$$h: [n] \rightarrow [n]$$

<!-- For example with 8 rows, the hash function could map them to:

$$[0, 1, 2, 3, 4, 5, 6, 7] \rightarrow [4, 1, 5, 6, 0, 2, 3, 7]$$ -->

Although it's not a necessary for the range of the hash values to be the same as the indices, let's assume for the sake of this example that it is. Then you can think of these permutations as, *where the row would have landed if we actually randomly shuffled the rows*. For example if we had some hash function $$h$$ and applied it to row index 4:

$$h(4) = 2$$

The way you can interpret this is, the row at position 4 got moved to position 2 after shuffling.

<!-- be generating a random permutation on the indices of the rows. We'll define functions that will take the index of a row as input and will output a random integer such that each row will have a unique integer associated with it. These kinds of functions are called, hash functions.   

What we're going to do instead is *implicitly* shuffle the rows by generating a permutation on the indices of the rows. In order to do this, we're going to introduce hash functions. -->

<!-- A hash function $$h$$ will map every index in the row to some unique integer. Although it's not a necessary for the range of the hash values to be the same as the indices, let's assume for the sake of this example that it is. Then you can think of these permutations as, *where the row would have landed if we actually randomly shuffled the rows*. For example with 8 rows, the hash function could map them to:

$$[0, 1, 2, 3, 4, 5, 6, 7] \rightarrow [4, 1, 5, 6, 0, 2, 3, 7]$$ -->

To simulate multiple iterations of implicit shuffling, we're going to apply multiple distinct hash functions $$h_{1}, h_{2}, ..., h_{k}$$ to each row index.

**Recipe for generating hash functions**

Pick a prime number $$p \ge m$$ where $$m$$ is the number of rows in the dataset. Then each hash function $$h_{i}$$ can be defined as:

$$h_{i}(x) = (a_{i}x + b_{i}) \mod p$$

Where $$a_{i}, b_{i}$$ are random integers in the range $$[1, p)$$ and $$[0, p)$$, respectively. The input $$x$$ to the function is the index of the row. To generate a hash function, all we have to do is pick the parameters $$a$$ and $$b$$.

For example, let's define three hash functions: $$h_{1}, h_{2}, h_{3}$$

<center>
$$h_{1}(x) = 7x \mod 11$$
$$h_{2}(x) = (x + 5) \mod 11$$
$$h_{3}(x) = (3x + 1) \mod 11$$
</center>

We'll be applying these hash functions to the rows of our toy dataset. Since the number of rows $$m = 8$$ is not a prime number we chose $$p = 11$$. As I've mentioned before, the values of the hash function need not be in the same range as the indices. As long as each index is mapped to a unique value, the range of the values actually makes no difference. If this doesn't make sense to you, let's unpack the example we have. In this case, our hash functions will produce values in the range $$[0, 10]$$. We can imagine expanding our dataset with a bunch of "null" type rows so that we have $$p=11$$ rows. We know that the "null" rows don't change the probability of two artists having the same signature, therefore our estimates should be be unaffected.

<!-- The reason for doing this is because we don't want to have collisions, that is we don't want more than one row to map to the same value for a given hash function. -->


<!-- But what this means is that our hash functions will produce values in the range $$[0, 10]$$, which is larger than your set of indices. This will actually end up not making any difference. To see why we can imagine expanding our dataset with a bunch of "null" type rows so that we have $$m=11$$. We know that the "null" rows don't change the probability of two artists having the same signature, therefore having a range bigger than the actual is not going to change anything. -->

<img src="../images/artist_matrix_hash.png" width="50%">

Each hash function defines an implicit shuffling order. As an exercise, for each hash function, iterate over the rows in the order that that hash function displays. For each column (artist) and each hash function store the index of the first-non zero element. Then to compute the Jaccard similarities, compare the stored values the same way we did before.

The MinHash algorithm is essentially doing the same thing but in a more efficient way by just doing a single pass over the rows.

<!-- Now that we have the hash functions, we're finally ready for the MinHash algorithm: -->

**MinHash**
```
Initialize all elements in the signature matrix sig to infinity
for each row r in the dataset
  Compute h_{i}(r) for all hash functions h_{1}, h_{2}, ..., h_{k}
  for each non-zero column c in the row r
    if sig(i, c) > h_{i}(r)
      update sig(i, c) = h_{i}(r)
```

When the algorithm terminates the signature matrix should contain all the minimum hash values for each artist and hash function pair.

The video below is an animation that simulates the algorithm over the toy dataset. Watching it should hopefully clear up any questions you have about how or why the algorithm works.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/qA4WdrY6aPk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

<!-- ## Further reading

- [Finding Similar Items](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf) - Chapter 3 of the book "Mining of Massive Datasets" by Jure Leskovec, Anand Rajaraman and Jeff Ullman. Has some really good exercises that are worth checking out.
- [Min Hashing](https://www.cs.utah.edu/~jeffp/DMBook/L4-Minhash.pdf) - Lecture notes from University of Utah CS 5140 (Data Mining) by Jeff M Phillips. Answers an important question that I have not addressed in this tutorial, "So how large should we set k so that this gives us an accurate measure?" -->

---

If you have any questions or you see any mistakes, please feel free to use the comment section below.
