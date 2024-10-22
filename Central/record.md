# Test outcome

## FedGNN

* ML-latest

  ```python
  TOTAL_CLIENTS = 610
  TRAIN_CLIENTS_RATIO = 0.7
  VALIDATION_RATIO = 0.2
  ROUND_CLIENTS = 128

  MINI_BATCH_SIZE = 32
  LR = 0.01
  INPUT_CHANNELS = 256
  HIDDEN_CHANNELS = 256
  USER_EMBEDDING_SIZE = 1 
  CLIP = 0.1

  ROUNDS = 50
  EPOCHS = 100


  INCLUDE_NEIGHBORS = True
  NEIGHBORS_THRESHOLD = 2

  SENSITIVITY = 5
  EPS = 1
  ```
* Without dp

  * Accuracy: 0.12661184
  * Loss:  1.1015505

    ![1713545735276](image/record/1713545735276.png)		![1713545743289](image/record/1713545743289.png)
* With-dp:

  * epsilon = 0.5
  * Accuracy:  0.03053064
  * Loss:  2.8827906
* ML-latest:

  ```python
  TOTAL_CLIENTS = 610
  TRAIN_CLIENTS_RATIO = 0.7
  VALIDATION_RATIO = 0.2
  ROUND_CLIENTS = 128

  MINI_BATCH_SIZE = 32
  LR = 0.01
  INPUT_CHANNELS = 256
  HIDDEN_CHANNELS = 256
  USER_EMBEDDING_SIZE = 1 
  CLIP = 0.1

  ROUNDS = 5
  EPOCHS = 50


  INCLUDE_NEIGHBORS = True
  NEIGHBORS_THRESHOLD = 2

  SENSITIVITY = 5
  EPS = 1

  # 100k Ratings Dataset
  RATINGS_DATAFILE = '../data/ml-latest-small/ratings.csv'
  MOVIES_INFO_DATAFILE = '../data/ml-latest-small/movies.csv'

  CUDA_VISIBLE_DEVICES=1
  ```

  * ```python
    #######################################################################################
    # Outcome
    epsilon = 0.5
    Final:
    Accuracy:  0.06325111 
    Loss:  1.9206945

    epsilon = 1
    Final:
    Accuracy:  0.20832632 
    Loss:  1.2171178

    epslon = inf
    Final:
    Accuracy:  0.11239206 
    Loss:  1.0717076

    ```

    * FedGNN-Fed-align

      ```python
      TOTAL_CLIENTS = 610
      TRAIN_CLIENTS_RATIO = 0.7
      VALIDATION_RATIO = 0.2
      ROUND_CLIENTS = 128

      MINI_BATCH_SIZE = 32
      LR = 0.01
      INPUT_CHANNELS = 128
      HIDDEN_CHANNELS = 256
      USER_EMBEDDING_SIZE = 1 
      CLIP = 0.1

      ROUNDS = 5
      EPOCHS = 30


      INCLUDE_NEIGHBORS = True
      NEIGHBORS_THRESHOLD = 2

      ###############################################################
      # Share operation
      SHARE_USER_RATIO = 0.5
      SHARE_HISTORY_RATIO = 0.5

      SENSITIVITY = 5
      EPS = 1

      NEI_LEN = 100

      # 100k Ratings Dataset
      RATINGS_DATAFILE = '../data/ml-latest-small/ratings.csv'
      MOVIES_INFO_DATAFILE = '../data/ml-latest-small/movies.csv'

      CUDA_VISIBLE_DEVICES=1
      ```

      * Result:
        1. SHARE_USER_RATIO = 0.5/SHARE_HISTORY_RATIO = 0.5

           Final:
           Accuracy:  0.09060073
           Loss:  1.050204
        2. Final:
           Accuracy:  0.105318025
           Loss:  0.9780921
        3. Final:
           Accuracy:  0.089892395
           Loss:  1.0470896
        4. Final:
           Accuracy:  0.10979281
           Loss:  1.056239
      * 对比结果
        1. Final:
           Accuracy:  0.101373
           Loss:  1.1208436
        2. fed-align-retrained:Final:
           Accuracy:  0.139453
           Loss:  1.0352383
        3. Accuracy:  0.24288386
           Loss:  0.9858942
  * 超参数：
    0.5-0.1：Loss:  1.2173642 1.3778411
    0.5-0.2  Loss:  1.2202846 Loss：1.43512
  * 0.5-0.3 Loss:  1.0593351 Loss:  1.0172594
  * 0.5-0.4 Loss:  1.0520182 Loss: 1.1084726
  * 0.5-0.6 Loss:1.002079 Loss:1.0771383
  * 0.5-0.7 Loss: 1.0498922 0.98525274
  * 0.5-0.8loss:1.0940604 1.1253966
  * Loss:  1.0940604 1.0407678
  * 0.9954345 0.202749
    Loss:  1.168987
  * 0.1-0.5

0.1-0.9：Final:
Accuracy:  0.11633526
Loss:  1.0120218
