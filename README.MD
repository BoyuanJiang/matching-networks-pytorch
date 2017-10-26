This is the pytorch implement of **[Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)**
![architecture](https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/architecture.png)
## Train 
If you want to train the model,simply run the code
```
python mainOmniglot.py 
```

You can set *fce = True* if you want use Full Context Embeddings.You can use the follow set in mainOmniglot.py for a 5-way one shot learning.
```
classes_per_set = 5
samples_per_class = 1
```

## Result
After about 30 epoches,you can achieve about 97% accuracy on train set and 96% on val and test set.
![result](https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/result.png)

## Acknowledgement
This work refer to [gitabcworld](https://github.com/gitabcworld/)'s implement,detail at[https://github.com/AntreasAntoniou/MatchingNetworks](https://github.com/AntreasAntoniou/MatchingNetworks).