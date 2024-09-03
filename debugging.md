# Debugging

I am currently working on debugging our feedforward neural network model in order to correct its functionality.
The debugging process is driving me downright [googledebunkers](https://youtu.be/pJmpA8JO4qE?si=kgpNBcinpYejKtSd&t=29). I have spent quite a while trying different solutions, but have had little success.

Currently debugging should be done on fnnCopySimplified.py. This file is a simplified version of the program which removes some unnecessary fluff.

The problem seems to be in the FNN model function.

## Desired behavior

The FNN function should return a Tensor object which indicates Alzheimer's diagnoses for each patient in the data batch (100/batch).

This may be better achieved in the format
predicted = ([[1, 0], [1, 0], [0, 1] ... [0, 1]])
where the index of the 1 indicates diagnosis (1 in index 0 = negative, 1 in index 1 = positive),
or in the format
predicted = ([1, 1, 0, 0, 1, ... 1])
where a 1 indicates a positive diagnosis and a 0 a negative diagnosis. I'm not sure which. I'll be thrilled with either, if it works.

## Actual behavior

I have been unable to get the FNN function to return an output which differentiates between inputs, much less provides correct diagnoses.
It will happily return a Tensor, but every result within will be the same. For example, I just got a return

outputs = ([[1,1], [1,1], [1,1], [1,1] ... [1,1]])

from both the function calls at line 87 and at line 100. This return will obviously be of no help diagnosing anybody. Sometimes the return will be 0s instead of 1s - still unhelpful.

This is why our accuracy peaks in the mid 60s% range. The number of negative diagnoses is about 64% of the total, so it makes sense on tests where the return is 0s.

## Attempted fixes

I have tried many approaches in my attempts to solve this problem. I started by trying to localize the problem; I'm reasonably convinced that it is in either the FNN function itself or in some call to it. But I could be off the mark with that assumption.
I've tried restructuring the function in a variety of ways, such as reordering calls to nn.Linear or nn.Sigmoid; I've tried removing a hidden layer to simplify things, then putting it back in hopes of something changing. I've tried changing loss functions and optimizer functions. I've tried resizing Tensors to make sure dimensions match up (I think this helped??).

I am Sisyphus and this is my boulder. Please help me get up this hill.
