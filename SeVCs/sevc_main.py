"""
Interface to VulDeePecker project
"""
import sys
import os
import pandas
import pickle 
from dataprocessing.clean_gadget import clean_gadget
from dataprocessing.vectorize_gadget import GadgetVectorizer
from models.blstm import BLSTM
from models.Transfomer import Transformer
from models.CNN import CNN
from models.RNN import RNN
import argparse
"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""
parser = argparse.ArgumentParser(prog='python3 sevc_main.py', description='NLP-based models for vulnerability detection using SeVC')
parser.add_argument('-m', '--model', help='DL model to train and evaluate, supported: cnn, rnn, bilstm, transformers', default='cnn', dest='model')
parser.add_argument('-dt', help='Input file consisting SeVC data for training and evaluating', dest='dt')
parser.add_argument('-df', help='Input file consisting vectors converted from SeVC', dest='df')
parser.add_argument('-emb', help='Embedding technique to covert SeVC/Code gadgets to vector. Supported: codebert, word2vec. Required when using raw SeVC or code gadget file, not required for processing vectors.', dest='emb') 

embedding_model_name = {
   'codebert': "microsoft/codebert-base",
   'word2vec' : 'word2vec'
}
device = "cpu"

args = parser.parse_args()
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 30 in line and gadget: 
                yield clean_gadget(gadget), gadget_val
                gadget = []
            elif stripped.split()[0].isdigit():
                if gadget:
                    # Code line could start with number (somehow)
                    if stripped.isdigit():
                        gadget_val = int(stripped)
                    else:
                        gadget.append(stripped)
            else:
                gadget.append(stripped)

"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
"""
def get_vectors_df(filename, embedding, modelname, vector_length):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(embedding, modelname, vector_length, device)
    #vectorizer = GadgetVectorizer(vector_length)
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vectorizer.add_gadget(gadget)
        row = {"gadget" : gadget, "val" : val}
        gadgets.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model " + embedding + "...", end="\r")
    vectorizer.train_model()
    print()
    vectors = []
    count = 0
    for gadget in gadgets:
        count += 1
        print("Processing gadgets...", count, end="\r")
        vector = vectorizer.vectorize(gadget["gadget"])
        row = {"vector" : vector, "val" : gadget["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    print("[+] Vectorizing SeVC completed.")
    return df
            
"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    if len(vars(args)) != 3:
        parser.print_help()
        exit()
        
    if (args.dt and args.df) or (not args.dt and not args.df):
        print("[-] ERROR: Only one type of input is supported, use dataframe file either SeVC file, not both or neither")
        parser.print_help()
        exit()
    if args.df != None:
        # DataFrame consting vectors exists
        print("[+] Vector file provided: " + args.df + ". Try reading the file...")
        vector_filename = args.df
        if os.path.exists(vector_filename):
            df = pandas.read_pickle(vector_filename)
        else:
            print("[-] ERROR: File not found")
            exit()
    elif args.dt != None:
        # DataFrame doesn't exist, start extracting vectors...
        print("[+] SeVC file provided: " + args.dt + ". Try reading the file...")
        filename = args.dt
        if os.path.exists(filename):
            parse_file(filename)
            base = os.path.splitext(os.path.basename(filename))[0]

            vector_length = 768
            vector_filename = base + "_gadget_vectors.pkl"
            if args.emb != None:
                if args.emb in embedding_model_name:
                    nmodel = embedding_model_name[args.emb]
                else:
                    print("[-] ERROR. Unsupported embedding technique")
                    exit()
            else:
                nmodel = 'word2vec'
            df = get_vectors_df(filename, args.emb, nmodel, vector_length)
            df.to_pickle(vector_filename)
        else:
            print("[-] ERROR: File not found")
            exit()

    model_name = args.model 
    print('----\n[+] Selected model:' + model)
    
    if model_name == 'cnn':
        model = CNN(df, name=base)
    elif model_name == 'bilstm':
        model = BLSTM(df, name=base)
    elif model_name == 'transformer':
        model = Transformer(df,name=base)
    elif model_name == 'rnn':
        model = RNN(df, name=base)
    else:
        print("[-] ERROR: Not supported model.")
        parser.print_help()
        exit()
        
    print('[+] Model training is starting...')
    model.train()
    print('[+] DONE. Start testing...')
    model.test()


    # # Mở file
    # with open('cwe119_cgd_gadget_vectors.pkl', 'rb') as f:
    #     # Đọc nội dung của file
    #     data = pickle.load(f)
    # print(data)
    
if __name__ == "__main__":
    main()
