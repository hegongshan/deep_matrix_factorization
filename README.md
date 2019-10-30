### Deep Matrix Factorization Models for Recommender Systems

A Non-official Implementation of "Deep Matrix Factorization Models for Recommender Systems"

See paper: http://www.ijcai.org/proceedings/2017/0447.pdf

If you use the codes for your paper as baseline implementation, please cite the link: https://github.com/hegongshan/deep_matrix_factorization 

### Environment Settings

We use Keras with Tensorflow as the backend.

- Keras version: 2.3.0
- TensorFlow: 2.0.0 

### Example to run the codes.

```
python dmf.py --dataset ml-1m --user_layers [512,64] --item_layers [1024,64] --epochs 100 --lr 0.0001
```

### Experimental Results
when epochs = 10 and lr = 0.001

|        | HR@10  | NDCG@10 | model file                                            |
|:------:|:------:|:-------:|:-----------------------------------------------------:|
| ml-1m  | 0.5225 | 0.2930  | model/ml-1m_u[512, 64]_i[1024, 64]_256_1572343913.h5  |   

**Tips**: Each epoch takes about an hour and a half.

If you are interested in DMF, you can try to set *lr* to 0.0001 and run 100 epochs. 

And then, HR@10 and NDCG@10 should be closer to the results in this paper.

Last Update: October 30, 2019 