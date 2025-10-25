Deep Learning 
=

Overview
--------

To gain a better understanding of deep learning, we're going to look
at deep averaging networks (DAN).  These are a very simple framework,
but they work well for a variety of tasks and will help introduce some
of the core concepts of using deep learning in practice.

In this homework, you'll use Pytorch to implement a DAN model for
determining the answer to a Quizbowl question.

You'll turn in your code on Gradescope. This assignment is worth 20 points.

Dataset
----------------

We're working with the same data as before, except this time (because
we need to use representations) we will need to create a vocabulary
explicitly (like we did for the earlier tf-idf homework).  However,
we'll give you that code. 

Keeping Things Simple
----------------

Although we'll use the usual Guesser class / setup, we're going to
keep things a little simpler.  


Pytorch DataLoader
----------------

In this homework, we use Pytorch's build-in data loader to do data
mini-batching, which provides single or multi-process iterators over the
dataset(https://pytorch.org/docs/stable/data.html).

The data loader includes two functions, `batchify()` and `vectorize()`. For
each example, we need to vectorize the question text into a vector using the 
vocabulary.  You don't need to implement anything here, but to implement the
rest of your code, you need to understand what they do.

What's the Loss Function?
----------------------

The first thing to understand is what objective we're optimizing.
When a question comes in, we turn it into a representation.  What's
our goal?  We want that representation to be closer to a question in
our train set with the correct label (answer / page) than questions
with different answers.

One way of doing that is by trying to predict what the final answer is 
by taking the prediction over answers and backpropagating into the 
answer representations.  This is the loss function required for this 
homework (*cross entropy*).

Another way of doing that is specifying the loss on the question representations
directly.  So if the wrong answer is closer, we push it away and pull the correct
answer closer.  You can implement this for extra credit.

In the code, the positive and negative examples are chose in the
``getitem`` function of the `QuestionData` class, but then turned into
matrices in the `batchify` function.  Walk through that code so you
understand everything.  Check the Pytorch documentation:

https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html 

Extreme Toy Data
----------------


The toy data are designed (and the unit tests use this) so that the words when
are on +1 / -1 on the y or x axis perfectly divide the data.

    def testEmbedding(self):
        for word, embedding in [["unk",      [+0, +0]],
                                ["capital",  [+0, -1]],
                                ["currency", [+0, +1]],
                                ["england",  [+1, +0]],
                                ["russia",   [-1, +0]]]:

This is because there are only four answers in the data, and the four words
combine to signal what the answer is.  After averaging the data, the four
quadrants represent the answer space.

    def testRealAverage(self):       
        reference = [([+0.5, +0.5], "england currency"),
                     ([-0.5, +0.5], "russia currency"),                     
                     ([-0.5, -0.5], "russia capital"),
                     ([+0.5, -0.5], "england capital")]

The provided network for testing for the final layer just stretches things out a bit.

    def testNetwork(self):
        embeddings = self.dan.dan_model.embeddings(self.documents)
        average = self.dan.dan_model.average(embeddings, self.length)
        representation = self.dan.dan_model.network(average)

        reference = [([+1.0, +1.0], "currency england"),
                     ([-1.0, +1.0], "currency russia"),                     
                     ([-1.0, -1.0], "capital russia"),
                     ([+1.0, -1.0], "capital england")]

Guide
-----

First, you need to check to make sure that you can construct an example from
text.  This is called "vectorizing" in the Pytorch pipeline.

    > python dan_test.py 
    Traceback (most recent call last):
    ======================================================================
    FAIL: test_train_preprocessing (__main__.DanTest)
    On the toy data, make sure that create_indices creates the correct vocabulary and
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/home/jbg/repositories/nlp-hw/dan/dan_test.py", line 155, in test_train_preprocessing
        self.assertEqual(guesser.vectorize(question), [3, 1])
    AssertionError: Lists differ: [0, 0] != [3, 1]

    First differing element 0:
    0
    3

    - [0, 0]
    + [3, 1]

Next, make sure that the network works correctly.  The unit tests define a
network that embeds the vocabulary and has two linear layers:

        embedding = [[ 0,  0],           # UNK
                     [ 1,  0],           # England
                     [-1,  0],           # Russia                     
                     [ 0,  1],           # capital
                     [ 0, -1],           # currency
                     ]

        first_layer = [[1, 0], [0, 1]] # Identity matrix
            
        second_layer = [[ 1,  1],        # -> London
                        [-1,  1],        # -> Moscow                        
                        [ 1, -1],        # -> Pound
                        [-1, -1],        # -> Rouble
                        ]

Those matrices are put into the parameters of the embeddings and linear layers:

        with torch.no_grad():
            self.toy_qa.linear1.bias *= 0.0
            self.toy_qa.linear2.bias *= 0.0
            self.toy_qa.embeddings.weight = nn.Parameter(torch.FloatTensor(embedding))
            self.toy_qa.linear1.weight.copy_(torch.FloatTensor(first_layer))
            self.toy_qa.linear2.weight.copy_(torch.FloatTensor(second_layer))

This should be a hint that you need to put these layers into a network of some sort!

After you've done that, the system should perfectly answer these questions
(e.g., that the "currency England" is the "Pound").  However, this is not the case at first:

    > python3 dan_test.py 
    Traceback (most recent call last):
      File "/home/jbg/repositories/nlp-hw/dan/dan_test.py", line 123, in testCorrectPrediction
        self.assertEqual(self.toy_dan_guesser.vectorize(words), indices)
    AssertionError: Lists differ: [0, 0] != [3, 1]

    First differing element 0:
    0
    3

    - [0, 0]
    + [3, 1]

Once you have the forward pass working with known weights, you'll need to train a network.

 The Actual Test Data
-------------------

For the training, the problem looks much the same, but you'll start from
random initialization and there will be lots of words that do not contribute
to finding the right answer.

The data are defined in guesser.py:

             "mini-train": [{"page": "Rouble", "text": "What is this currency of russia"},
                            {"page": "Pound", "text": "What is this currency of england"},
                            {"page": "Moscow", "text": "What is this capital of russia"},
                            {"page": "London", "text": "What is this capital of england"},
                            {"page": "Rouble", "text": "What 's russia 's currency"},
                            {"page": "Pound", "text": "What 's england 's currency"},
                            {"page": "Moscow", "text": "What 's russia 's capital"},
                            {"page": "London", "text": "What 's england 's capital"}],
             "mini-dev": [{"page": "Rouble", "text": "What currency is used in russia"},
                          {"page": "Pound", "text": "What currency is used in england"},
                          {"page": "Moscow", "text": "What is the capital and largest city of russia"},
                          {"page": "London", "text": "What is the capital and largest city of england"},
                          {"page": "Rouble", "text": "What 's the currency in russia"},
                          {"page": "Pound", "text": "What 's the currency in england"},
                          {"page": "Moscow", "text": "What 's the capital of russia"},
                          {"page": "London", "text": "What 's the capital of england"}],

The learned representations won't be as clean, but you should be able to get
perfect accuracy on this dataset.


```
python dan_guesser.py --question_source=gzjson --questions=./mini-train.json.gz --secondary_questions=./mini-dev.json.gz --limit=1000 --no_cuda --dan_guesser_max_classes=200 --dan_guesser_ans_min_freq=1
```

Scaling Up
-----------

We don't expect you to scale up to "real" data for this homework, but you can do so (particularly if you have a GPU).  For that, 

    python dan_guesser.py   --dan_guesser_hidden_units 50   --dan_guesser_vocab_size 30   --dan_guesser_max_classes 4   --dan_guesser_num_workers 0   --dan_guesser_num_epochs 100   --dan_guesser_embed_dim 50   --dan_guesser_nn_dropout 0.3   --dan_guesser_batch_size 4   --dan_guesser_criterion CrossEntropyLoss   --dan_guesser_device cuda --question_source=gzjson --questions=../data/qanta.guesstrain.json.gz --secondary_questions=../data/qanta.guessdev.json.gz --limit=10000


Then check to see how well the code does.

    > python eval.py --guesser_type=DanGuesser --question_source=gzjson --questions=../data/qanta.guessdev.json.gz --evaluate guesser --limit=10000 --no_cuda
    ...
    =================
    close 0.00
    ===================
    
                   guess: Pulsar
                  answer: Thornton_Wilder
                      id: 145775
                    text: The second act of a play by this man opens with a pair of speeches
                          offering the mottoes "Enjoy Yourselves" and "Save the Family". Food
                          poisoning-stricken actors, including Miss Somerset, have to be
                          replaced in the third act of that play by this man, which features a
                          maid named Lily Sabina and a member of the "Ancient and Honorable
                          Order of Mammals" who invents the wheel. A dead woman attempts to
                          relive her (*) twelfth birthday in another of this man's plays, whose
                          cast includes the alcoholic choir director Simon Stimson. The Antrobus
                          family survives an ice age in one of his plays, while the Stage
                          Manager officiates the wedding of Emily Webb and George Gibbs in
                          Grover's Corners in another. For 10 points, name this playwright of
                          The Skin of Our Teeth and Our Town.
    --------------------
    =================
    hit 0.00
    ===================
    
                   guess: Surface_tension
                  answer: Surface_tension
                      id: 145843
                    text: Griffith's criterion sets the square root of the product of Young's
                          modulus and this quantity for a solid equal to a constant to determine
                          if a material fractures. The pressure differential inside a bubble
                          equals four times this quantity over the radius of the bubble.
                          Electrowetting is used when this quantity is high in order to decrease
                          the contact angle. This quantity equals the increase in Gibbs energy
                          per increase in (*) exposed area. If this quantity is negative, then a
                          liquid in a barometer forms a concave meniscus. This value, which is
                          given in dynes per centimeter and symbolized either sigma or gamma,
                          leads to capillary action. For 10 points, name this quantity which is
                          positive when cohesion is stronger than adhesion, and causes molecules
                          at an interface to cling to each other.
    --------------------
    =================
    Precision @1: 0.0009 Recall: 0.0018

Because many of you don't have GPUs, our goal is not to have you train a
super-converged model.  We want to see models with a non-zero recall and
precision guess over at least hundreds of possible answers.  It doesn't have to be
particularly good (but you can get more extra credit if you invest the time).


What you have to do
----------------

**Coding**: (20 points)
1. Understand the structure of the code, particularly the
   `QuestionData` class.
2. Write the data `vectorize()` funtion.
3. Write DAN model initialization. 
3. Write the `average()` function.
4. Write model `forward()` function.


Pytorch install
----------------
In this homework, we use Pytorch.  

You can install it via the following command (linux):
```
conda install pytorch torchvision -c pytorch
```

If you are using MacOS or Windows, please check the Pytorch website for installation instructions.

For more information, check
https://pytorch.org/get-started/locally/.

Extra Credit
----------------

The preferred extra credit for this homework is using a ranking-based loss function.
Most of the code for finding positive and negative examples is already provided, but
you may need (or want) to tweak the code so that it gives you what you want.

There are lots of other things you could do for extra credit, but here are
some ideas:

* Initialize the word representations with
word2vec, GloVe, or some other representation.  Compare the final performance
based on these initializations *and* see how the word representations
change. Write down your findings in analysis.pdf.

* Have the dropout depend on the index of words so that later text is
  more likely to disappear.  This will make it work better on
  pyramidal questions.
  
* Select the negative example more intelligently than randomly (e.g.,
  pick an example that looks similar based on tf-idf but has a
  different label).  Or refresh the negative examples based on the
  model errors.  
  
* Form the vocabularly more intelligently (e.g., put "this Finnish
  composer" into a single word) so that word order can have a bit more
  help to the model. [Suggestions: Use Spacy's ``noun_chunks``
  function after running an ``nlp`` analysis.

You can also get extra credit by getting the highest precision and recall by
tuning training parameters.  If you have other ideas, just ask, and we
can say whether your proposal makes sense.

Good Enough
------------

To get full points on this assignment, you'll need to have an implementation that can get perfect on the `mini-dev` dataset when trained on the `mini-train` dataset (6 points). Also, you'll need to pass all test cases in `dan_test.py` (14 points).

What to turn in 
----------------
Please make sure that you get model pickles named correctly, or our autograder won't detect and you may lose the points.

**Good Enough**: (20 points)
1. Submit your `dan_guesser.py` and `parameters.py` file.
2. Upload your model that train using `mini-train.json.gz` under name `dan_main.torch.pkl`
3. (optional) `dan_main.data.pkl`  (if you train with MarginRankingLoss)

**Extra Credit**: (5 points)
1. Submit your `analysis.pdf` file to describe what you've done, either by experimenting ranking-based loss, word representations, parameter tuning, etc. (Please make sure that this is **PDF** file!      No more than one page, include your name at the top of the pdf.)
2. Upload your model that train using `qanta.guesstrain.json.gz` under name `dan_ec.torch.pkl`
3. (optional) `dan_ec.data.pkl`  (if you train with MarginRankingLoss)
4. (Optional) Upload the wordvectors you use.

FAQ
----

*Q:* Why is my accuracy zero?

*A:* The first thing to check is that you've implemented everything correctly.  If you're passing the unit tests, you can correctly learn from the toy data, and your gradients are non-zero, you're probably okay.

The next thing to think about is how many answers your system has.
I.e., what is the size of the examples that it's training on.  If it's
too small (i.e., your system can't give many answers, the accuracy is
going to be low).  If it's too large, your model might not have the
representational power to find closest questions.

The thing is, the number of answers your system can provide is
determined by your training data.  The code is set up to only use
answers that have at least `--DanGuesser_min_answer_freq` questions
associated with them.  So if your training set is too small, there
won't be enough answers and your accuracy will always be low.  Another
issue is that if you have too few answers, most of the answers will be
unknown (they all get mapped into one answer).  So your system will
always guess the uknown answer.  So you may want to downsample how
many of the unknown examples you train on with `--DanGuesser_unk_drop`
(1.0 will remove all of the unknown answers).

*Q:* There aren't enough answers or too many!  What can I do?

*A:* Look at the DanGuesser_min_answer_freq flag to adjust what answers you include.

*Q:* Too many of the answers are unknown!  What can I do?

*A:* Look at the DanGuesser_unk_drop flag to adjust how many "unknown" examples you keep.

