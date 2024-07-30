# Instruction to use Defect-Scanner with SeVCs as input
## Prepare dataset of SeVCs
Defect-Scanner supports two different forms of code gadgets to be used as input.
1. SeVCs extracted from C/C++ programs using the method mentioned in [SySEVR Github](https://github.com/SySeVR/SySeVR/tree/master), with a corresponding label of 0 or 1.
2. Vectors which are tokenized SeVCs using supported embedding techniques.
An example of `.pkl` file (Python pickled object) consisting of vectors extracted from source code is available in `data` folder (~11GB unzipped).

## Running Defect-Scanner with prepared dataset
Run `sevc_main.py` using Python3 and provide arguments to specify the path to the dataset (use `-dt` and `-df` options to use the dataset in Option 1 or Option 2 above, respectively) as well as the desired embedding technique and DL model.
```
python3 sevc_main.py [-h] [-dt <path_to_txt_file>] [-df <path_to_vector_file>] [-m MODEL] [-emb EMBEDDING]

-h, --help                  Print this message and exit
-dt <path_to_txt_file>      Input file consisting code gadget data for training and evaluating
-df <path_to_vector_file>   Input file consisting vectors converted from code gadgets
-m|--model MODEL            (Optional) DL model to train and evaluate. Supported DL models: cnn, rnn, blstm, transformers. Default: cnn.
-emb EMBEDDING              (Required when using -dt) Supported embedding techniques: word2vec, codebert. Default: word2vec.

