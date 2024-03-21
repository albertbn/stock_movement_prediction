### How to run the project

For the best result using CNN, run:
```
from TrainTestLoopCnn import TrainTestLoopCnn

with TrainPredictCnn() as obj:
    obj.run_train()
```

This repository is loaded in a Jupyter notebook here, where additional charts and statistics are displayed:
https://colab.research.google.com/drive/1lulfVf3qYmYmFsWWL7R60amE9wwXXf0l?usp=sharing


### Dataset
The dataset is saved as a pandas feather dataset here:<br>
https://getfiles.adcore.com/img/yumoxu_text_embeddings.feather

The notebook that created the above dataset is here:
https://colab.research.google.com/drive/11iTOVoLCEjo34eRwkNIIcAorVQU2-a2S?usp=sharing

### Links and additional info

the dataset in papers with code + 25 papers mentioning this dataset<br>
https://paperswithcode.com/dataset/stocknet-1


the github to download the dataset<br>
https://github.com/yumoxu/stocknet-dataset


a pytorch transformer model for predicting stock prices with prices and tweet data<br>
https://www.datasciencebyexample.com/2023/05/15/stock-price-prediction-using-transformer-in-pytorch/


a github .ipynb from the above blog with the code for the temporal/timeline based transformer model<br>
https://github.com/robotlearner001/blog/blob/main/stock-price-prediction-using-transformer-toy-example/2023-05-15-stock%20price%20prediction%20using%20transformer%20in%20pytorch.ipynb

![r-squared](https://getfiles.adcore.com/img/r-squared.png)

https://www.quora.com/How-do-we-reconcile-this-fact-%E2%80%9CThe-mean-squared-error-in-regression-is-close-to-zero-yet-the-R-squared-values-are-much-lower%E2%80%9D
