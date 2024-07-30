# Instruction to use Defect-Scanner with Code gadgets as input
## Prepare dataset of Code gadgets
Defect-Scanner supports two different forms of code gadgets to be used as input.
1. Code gadgets in the form of extracted C/C++ code with a corresponding label (0 or 1) according to their vulnerability as in [VulDeePecker dataset](https://github.com/CGCL-codes/VulDeePecker). Below are 2 examples of code gadgets, with the final integer of 0 being their labels.
```
1 CVE-2010-1444/vlc_media_player_1.1.0_CVE-2010-1444_zipstream.c cfunc 449
ZIP_FILENAME_LEN, NULL, 0, NULL, 0 )
char *psz_fileName = calloc( ZIP_FILENAME_LEN, 1 );
if( unzGetCurrentFileInfo( file, p_fileInfo, psz_fileName,
vlc_array_append( p_filenames, strdup( psz_fileName ) );
free( psz_fileName );
0
---------------------------------
2 CVE-2010-1444/vlc_media_player_1.1.0_CVE-2010-1444_zipstream.c cppfunc 449
char *psz_fileName = calloc( ZIP_FILENAME_LEN, 1 );
ZIP_FILENAME_LEN, NULL, 0, NULL, 0 )
if( unzGetCurrentFileInfo( file, p_fileInfo, psz_fileName,
vlc_array_append( p_filenames, strdup( psz_fileName ) );
free( psz_fileName );
0
---------------------------------
```
2. Vectors which are tokenized code gadgets using supported embedding techniques.
An example of `.txt` file consisting of code gadgets is available in `data` folder (>400MB unzipped).

## Running Defect-Scanner with prepared dataset
Run `codegadget_main.py` using Python3 and provide arguments to specify the path to the dataset (use `-dt` and `-df` options to use the dataset in Option 1 or Option 2 above, respectively) as well as the desired embedding technique and DL model.
```
python3 codegadget_main.py [-h] [-dt <path_to_txt_file>] [-df <path_to_vector_file>] [-m MODEL] [-emb EMBEDDING]

-h, --help                  Print this message and exit
-dt <path_to_txt_file>      Input file consisting of code gadget data for training and evaluating
-df <path_to_vector_file>   Input file consisting of vectors converted from code gadgets
-m, --model MODEL            (Optional) DL model to train and evaluate. Supported DL models: cnn, rnn, blstm, transformers. Default: cnn.
-emb EMBEDDING              (Required when using -dt) Supported embedding techniques: word2vec, codebert. Default: word2vec.

