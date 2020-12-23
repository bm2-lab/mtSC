# Integrating multiple references for single cell assignment
## Install
1. Clone the repository
```
git clone https://github.com/bm2-lab/mtSC.git  
```
2. Install the dependencies
```
pip install mtSC
```
3. Create folder  
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
## Data preprocessing
A routine normalization and quality control should be performed. There are three criteria: the number of genes detected (default >500), the number of unique molecular identifiers induced (default >1500), and the percentage of mitochondrial genes detected (default <10% among all genes). Only cells satisfying all three criteria are retained to construct the reference data. Then all datasets were normalized with commonly used method, i.e., scaling to 10000 and then with log(counts+1). Next, rare cell types whose cell number is less than 10 should be removed because such cell types do not contain enough information and are unreliable for subsequent assignment.
## Format of input data
The format of training data should be a csv or tab-delimited txt format where the columns correspond to genes and the rows correspond to cells. The column of cell types should be the last column and named as "cell_label". A sample file looks something like:

|   | tspan6 | dpm1 | cell_label |
| ------------- | ------------- |------------- | ------------- |
| pbmc1_SM2_Cell_133  | 0.745639  |0.0  |CD4+_T_cell |
| pbmc1_SM2_Cell_142  | 0.0  |0.778851  |B_Cell  |

The format of test data is the format of training data without the column of "cell_label". We also put example data in https://www.jianguoyun.com/p/DbqGHM0Q7oftCBiFjNQD. You can download whole `train_set` and `test_set` folder. Once Downloaded, you need to unzip the `train_set.zip` and `test_set.zip` and then  use the unzipped `train_set` folder and `test_set` folder to replace the origin `train_set` folder and `test_set` folder. The demo example can be run by calling the following command.
```
python run.py
```
## How mtSC works
### Cell assginment with model trained by your datasets
You just need to put reference datasets files in the `train_set` folder, put the query datasets files in the `test_set` folder and run the following command.
```
python run.py
```
### Cell assginment with pre_trained models
You can directly use models we have trained. Models can be downloaded through https://www.jianguoyun.com/p/DbZJv8QQ7oftCBiPg9QD. Once Downloaded, you need to unzip the `pre_trained.zip`  and use the unzipped `pre_trained` folder to replace the origin `pre_trained` folder. There are four trained models--'brain','immune1','immune2','pancreas'. You can view `cell_type.txt`  to select the model satisfying your need. For illustration purpose, we took 'brain' model as an example. You just need to put the query dataset file in the `test_set` folder and run the following command.
```
python test_with_trained_model.py -m brain
```

 ## Check the results
 Results are in the `result.txt`  after running.
## Citation  
Qi Liu, et al. Integrative single cell assignment with multiple references by Multi-task Deep Metric Learning
. 2020 (Manuscript submitted)  
## Contacts  
csq_@tongji.edu.cn or qiliu@tongji.edu.cn
