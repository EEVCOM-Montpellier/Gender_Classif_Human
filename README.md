# Human Gender Classification with CFD

The objective of this tool is to predict femininity of Human Faces based on the probability to classify an image in "Women" category and on the distance to male-centroid or female-centroid.


## Theory
For the classification task, we use transfer learning method (architecture VGGFace) on a learning set.
![30% center](/img/meth1.png)

We predict the femininity on a testing set.
![30% center](/img/meth2.png)

The task is repeated for each image of the dataset. (597 times)

## Requirements

* Install directly all requirements here (in your env ; ex py35):

```
pip install -r requirements.txt
```


* If using a virtual env (with Anaconda) please create an environment
https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc


* If using PC ES10SA3
1. Using GPU
```
activate tf_gpu
```
2. Using GPU
```
activate tf_cpu
```

* Model from rc_malli repo

```
pip install keras_vggface
 ```

 * Database from CFD: https://chicagofaces.org/default/

## Run

Please launch the tool at base.

Run the program
```
 python main.py
```

 The tool will create a folder inside `results/` a folder with the current date. (ex: `results/2019-04-30-19-28`)  
 Inside this folder :  
 * `results/2019-04-30-19-28/all_fc6_fc7` with a csv file fo each testing images (ex : `log_AF-201` with the activation of the testing image AF-201 (first row) and the learning images for the separate layers fc6 and fc7 )

 * `results/2019-04-30-19-28/all_logs_csv`  with a csv file fo each testing image to track the evolution of loss and accuracy over epochs

  * `results/2019-04-30-19-28/all_fc6_fc7` with a csv file fo each testing image with weights of each neurons for layers fc6 [0], fc7 [1], fc6+fc7 [2]

  * `all.performances.txt` performances for each model for each testing images

  * `log_file.txt` tracking

  * `params.txt` file with parameters used

  * `tags_pict_pred_tmp.txt`: temporary file with dataframe partially filled (/!\ if the loop stop, this allow us to keep results for some images)

  * `tags_pict_pred.txt`: final dataframe


## Scripts
* **main.py** : main script to obtain femininity for each image with different measures of femininity (centroid distance and classification probabilities with svm, sigmoid)
* **make_dataset.py** : formate meta-datas for images, you can use it to formate your data but you can use directly the meta-datas  `datas/pict_metadatas`



## More details:
Contact
 [soniamaitieo](https://github.com/soniamaitieo)
