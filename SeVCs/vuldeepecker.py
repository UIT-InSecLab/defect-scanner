"""
Interface to VulDeePecker project
"""
import sys
import os
import pandas
import pickle 
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
from blstm import BLSTM
from Transfomer import Transformer
from CNN import CNN
from RNN import RNN

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""

model_name = "microsoft/codebert-base"
device = "cpu"

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
def get_vectors_df(filename, vector_length):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(model_name, vector_length, device)
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
    print("Training model...", end="\r")
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
    return df
            
"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    if len(sys.argv) != 2:
        print("Usage: python vuldeepecker.py [filename]")
        exit()
    filename = sys.argv[1]
    parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]

    vector_filename = base + "_gadget_vectors.pkl"

    vector_length = 768
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
        df.to_pickle(vector_filename)

    #print('---------------------------------------------')
    #print("BiLSTM")
    # BiLSTM
    #blstm = BLSTM(df, name=base)
    #blstm.train()
    #blstm.test()
    
    #print("Transformer")
    # # Transfomer
    #transformers = Transformer(df, name=base)
    #transformers.train()
    #transformers.test()
    
    #print('---------------------------------------------')
    #print("CNN")
    # # CNN
    #cnn = CNN(df, name=base)
    #cnn.train()
    #cnn.test()
    
    print('---------------------------------------------')
    print("RNN")
    # # RNN
    rnn = RNN(df, name=base)
    rnn.train()
    rnn.test()


    # # Mở file
    # with open('cwe119_cgd_gadget_vectors.pkl', 'rb') as f:
    #     # Đọc nội dung của file
    #     data = pickle.load(f)
    # print(data)
if __name__ == "__main__":
    main()
