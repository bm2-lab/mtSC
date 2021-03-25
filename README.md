# mtSC: Integrating multiple references for single cell assignment
## Introduction
mtSC is a novel, flexible and generalized multitask deep metric learning-based framework for single cell assignment based on multiple references. Previous strategies for single-cell assignment with multiple references rely on data- or decision-level integration, while limitations remain. Different from the previous strategies, mtSC regards each reference dataset as a task, and different tasks can be complementary to improve single-cell assignment, while the overcorrection of the batch effect can be avoided. Such a novel integration strategy provides a flexible and reliable way to integrate related reference datasets. On the other hand, two additional advantages of mtSC were proven in this study: (1) mtSC performs increasingly better as the number of reference datasets increases, and (2) mtSC enables cross-species single-cell assignment, especially for specific tissues with very few sequencing datasets available for a specific species. These two characteristics of mtSC are of great potential utility when much more sequencing data on different species have accumulated in the future.
## Workflow
![](https://github.com/bm2-lab/mtSC/blob/master/mtSC_workflow.jpg)
mtSC comprises two main steps: model learning and cell assignment.
* (1) In the model learning process of mtSC, each dataset is considered a single task, and a corresponding loss is calculated. All the losses are added together and utilized to update the model parameters through a backpropagation algorithm, then the parameter-shared deep metric learning network (PS-DMLN) was trained for the next cell assignment process.
* (2) In the cell assignment process of mtSC, the trained PS-DMLN is utilized to transform the query cells. Then, the transformed query cells are compared against transformed reference cells, and the predicted cell type with the highest similarity among all the transformed reference datasets is obtained.

## Install
Environment: Python>=3.6
* (1) Clone the repository
```
git clone https://github.com/bm2-lab/mtSC.git  
```
* (2) Install the dependencies
```
pip install mtSC
```
* (3) Create folder 

   Linux or Mac:
   ```
   cd mtSC
   sh create_folder.sh
   ```
   Windows:
   ```
   cd mtSC
   create_folder.bat
   ```
## Tutorial
### Format of input data
A routine normalization and quality control should be performed. For example, there are three commonly used cell quality criteria, namely, the number of genes detected (default >500), the number of unique molecular identifiers induced (default >1500), and the percentage of mitochondrial genes detected (default <10% among all genes). Then, datasets should be normalized, i.e., scaling to 10,000 and then with log(counts+1).
The format of training data should be a csv or tab-delimited txt or h5ad(scanpy) format where the columns correspond to genes and the rows correspond to cells. The column of cell types should be the last column and named as "cell_label". In a word, the format of training data is a transposed normalized dataset with a cell type column in the right. A sample file looks something like:

|   | tspan6 | dpm1 | cell_label |
| ------------- | ------------- |------------- | ------------- |
| pbmc1_SM2_Cell_133  | 0.745639  |0.0  |CD4+_T_cell |
| pbmc1_SM2_Cell_142  | 0.0  |0.778851  |B_Cell  |

The format of test data is the format of training data without the column of "cell_label".
We also provide a script `preprocess.py` to handle the cell quality control and normalization of origin counts matrix dataset. The processed file will end up with "\_treated.h5" in its name and can read through `pandas` package's `read_hdf` function. After processing by the script, you just need to add the cell type column in the right of the processed matrix. The origin counts matrix dataset should be a csv or tab-delimited txt or h5ad(scanpy) format where the columns correspond to cells and the rows correspond to genes looks like:
|   | pbmc1_SM2_Cell_133 | pbmc1_SM2_Cell_142 |
| ------------- | ------------- |------------- |
| tspan6  | 4  |0  |
| dpm1  | 0  |9  |
The script can be run by calling the following command. The `filename` is filename of the origin counts matrix dataset.
  ```
  python preprocess.py -f filename
  ```
### How mtSC works
* **Example datasets:** The example data is in https://www.jianguoyun.com/p/DbqGHM0Q7oftCBiFjNQD. You can download whole `train_set` and `test_set` folder. Once downloaded, you need to unzip the `train_set.zip` and `test_set.zip` and then use the unzipped `train_set` folder and `test_set` folder to replace the origin `train_set` folder and `test_set` folder. The demo example can be run by calling the following command.
  ```
  python run.py
  ```
* **Cell assginment with model trained by your datasets:** You just need to put reference datasets files in the `train_set` folder, put the query datasets files in the `test_set` folder and run the following command.
  ```
  python run.py
  ```
* **Cell assginment with pre_trained models:** You can directly use models we have trained. Models can be downloaded through https://www.jianguoyun.com/p/DbZJv8QQ7oftCBiPg9QD. Once downloaded, you need to unzip the `pre_trained.zip` and use the unzipped `pre_trained` folder to replace the origin `pre_trained` folder. There are four trained models--'brain','PBMC-Ding','PBMC-Mereu','pancreas'. You can view `cell_type.txt`  to select the model satisfying your need. For illustration purpose, we took 'brain' model as an example. You just need to put the query dataset file in the `test_set` folder and run the following command.
  ```
  python test_with_trained_model.py -m brain
  ```
  Or if you have trained your own model by running `run.py`. You can also use your own model by
  ```
  python test_with_trained_model.py -m new_model
  ```
* **Check the results:** For all the above tests, the assignment results can be found in the `result.txt`  after running.
## Citation  
Integrating multiple references for single-cell assignment. 2021 (Manuscript submitted)  
## Contacts  
csq_@tongji.edu.cn, bioinfo_db@163.com or qiliu@tongji.edu.cn
