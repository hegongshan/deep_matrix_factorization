### Deep Matrix Factorization Models for Recommender Systems

A non-official Implementation of "Deep Matrix Factorization Models for Recommender Systems"

See paper: http://www.ijcai.org/proceedings/2017/0447.pdf

## Environment Settings

We use Keras with Tensorflow as the backend.

- Keras version: 2.3.0
- TensorFlow: 2.0.0 

### Example to run the codes.

```
python dmf.py --dataset ml-1m --user_layers [512,64] --item_layers [1024,64] --epochs 100
```

### Experimental Results
when epochs = 10

|        | HR@10  | NDCG@10 | model file                                            |
|:------:|:------:|:-------:|:-----------------------------------------------------:|
| ml-1m  | 0.4505 | 0.2391  | model/ml-1m_u[512, 64]_i[1024, 64]_256_1571999909.h5  |   

**Tips**: Each epoch takes about an hour and a half.

If you are interested in DMF, you can try to run 100 epochs. 

And then, HR@10 and NDCG@10 should be closer to the results in this paper.

Last Update: October 26, 2019 