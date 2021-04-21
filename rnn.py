import numpy as np
import math
from collections import defaultdict


"""
Notations : 
x : single training example 
y : single actual output corresponding to X
X : batch of training examples
Y : batch of actual outputs corresponding to X
"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def create_vocab_for_input_text(input_text,vocab_size=50):
    vocab=defaultdict(int)
    f=open(input_text)

    for line in f:

        for word in line.split():
            vocab[word.lower()]+=1

    vocab=[temp[0] for temp in sorted(vocab.items(),key=lambda x:x[1],reverse=True)[:vocab_size-1]]
    vocab=["unk"]+vocab

    vocab_to_int=dict([(w,i) for i,w in enumerate(vocab)])
    int_to_vocab=dict([(i,w) for i,w in enumerate(vocab)])

    return vocab_to_int,int_to_vocab

def create_training_examples_from_corpus(file):
    """
    :param file: Raw text file
    :return: List of x and list of y , where x is a sentence and y is actual output corresponding to x.

    Exa :
    x : bangalore is beautiful city
    y:  is beautiful city "unk"
    """
    f=open(file)
    x_training=[]
    y_training=[]

    for line in f:

        x=[word.lower().strip() for word in line.split()]
        #add next word as actual output
        y=[word.lower().strip() for word in line.split()[1:]]+["unk"]

        x_training.append(x)
        y_training.append(y)
    return x_training,y_training

def create_one_hot_encoding_for_sen(x,y,vocab_to_int,int_to_vocab,T=3):

    """
    :param x: sample input sentence and 
    :param y: actual output corresponding to x
    :param vocab_to_int : vocab to int dictionary
    :param int_to_vocab : int to vocab dictionary
    :return: one hot encoded x and y

    Exa :
    x : ["bangalore","is","beautiful","city"]
    y:  ["is","beautiful","city","unk"]
    vocab_to_int: {"bangalore":1,"is":2,"unk":0}
    int_to_vocab  {1:"bangalore",2:"is",3:"unk}
    """
    
    x_one_hot_encode = list()
    y_one_hot_encode = list()


    #convert each word of senetence into integer
    x_integer_encoded = [vocab_to_int[word.lower()] if word in vocab_to_int else vocab_to_int["unk"] for word in
                       x][:T]

    #interger encoding for actual words
    y_integer_encoded=[vocab_to_int[word.lower()] if word in vocab_to_int else vocab_to_int["unk"] for word in
                       y][:T]


    #padding if sequence len is less then max_sen_length
    x_integer_encoded=x_integer_encoded+[0]*(T-len(x_integer_encoded))
    y_integer_encoded=y_integer_encoded+[0]*(T-len(y_integer_encoded))

    vocab=int_to_vocab.keys()
    for value in x_integer_encoded:
        temp=[0]*len(vocab)
        temp[value]=1
        x_one_hot_encode.append(temp)

    for value in y_integer_encoded:
        temp=[0]*len(vocab)
        temp[value]=1
        y_one_hot_encode.append(temp)

    return np.array(x_one_hot_encode),np.array(y_one_hot_encode)

class Rnn(object):
    def __init__(self,vocab_to_int,int_to_vocab,emb_dim_size=30,vocab_size=50,hidden_size=10):
        """
        Initialization function for RNN parameters 
        :param vocab_to_int: vocab to int dictionary
        :param int_to_vocab: int to vocab dictionary
        :param emb_dim_size: embedding dimensiona
        :param vocab_size: vocab size for input text
        :param hidden_size: hidden deimension
        """
        
        self.emb_dim_size=emb_dim_size
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size

        self.E=np.random.uniform(-math.sqrt(6)/math.sqrt(self.emb_dim_size+self.vocab_size),
                                 math.sqrt(6)/math.sqrt(self.emb_dim_size+self.vocab_size),(self.emb_dim_size,self.vocab_size))
        self.Wh=np.random.uniform(-math.sqrt(6)/math.sqrt(self.hidden_size+self.hidden_size),
                                  math.sqrt(6)/math.sqrt(self.hidden_size+self.hidden_size),(self.hidden_size,self.hidden_size))
        self.We=np.random.uniform(-math.sqrt(6)/math.sqrt(self.hidden_size+self.emb_dim_size),
                                  math.sqrt(6)/math.sqrt(self.hidden_size+self.emb_dim_size),(self.hidden_size,self.emb_dim_size))
        self.U=np.random.uniform(-math.sqrt(6)/math.sqrt(self.vocab_size+self.hidden_size),
                                 math.sqrt(6)/math.sqrt(self.vocab_size+self.hidden_size),(self.vocab_size,self.hidden_size))
        self.b1=np.random.uniform(-math.sqrt(6)/math.sqrt(self.hidden_size+1),
                                  math.sqrt(6)/math.sqrt(self.hidden_size+1),(self.hidden_size,))
        self.b2=np.random.uniform(-math.sqrt(6)/math.sqrt(self.vocab_size+1),
                                  math.sqrt(6)/math.sqrt(self.vocab_size+1),(self.vocab_size,))

        self.int_to_vocab=int_to_vocab
        self.vocab_to_int=vocab_to_int


    def forward_propagate(self,x):

        """
        Run forward propagation for a given input x (hot encoded)
        :param T: Maximum Word length in a given input sentence
        :param x: list of  one hot vectors for each word in sentence ,length of list will be equal to T
        :return:list of o and h, where
                o - list of list . Each entry is list containing predicted probabilities for all vocab words at time t
                h- list of list. Each entry is hidden vector at time t.
        """
        T=x.shape[0]
        o=np.zeros((T,self.vocab_size),dtype=np.float)
        h=np.zeros((T+1,self.hidden_size),dtype=np.float)
        h[-1]=np.random.rand(self.hidden_size,) #dim=Dh

        e = np.matmul(x, self.E.T)  # dim (T,d)
        assert e.shape == (T, self.emb_dim_size)

        for t in range(T):
            z=np.matmul(h[t-1],self.Wh.T)+np.matmul(e[t],self.We.T)+self.b1 #dim Dh
            assert z.shape[0]==(self.hidden_size)

            h[t]=sigmoid(z) #dim Dh
            assert h[t].shape[0]==(self.hidden_size)

            o[t]=softmax(np.matmul(h[t],self.U.T)+self.b2) #dim V

        return [o,h]


    def backpropogation(self,x,y):

        """
        Run backward propagation for a given input x (hot encoded) and y
        :param x: Single input of dimention (T,v) , where Time steps to be considered, and v size of vocab
        :param y: Actual output of dimention (T,v), where Time steps to be considered, and v size of vocab
        :param T: Time steps to be considered, Default is 3
        :return: gradients of parameters we,wh and u
        """
        dldu=np.zeros_like(self.U)
        dldwe=np.zeros_like(self.We)
        dldwh=np.zeros_like(self.Wh)
        dldE=np.zeros_like(self.E)

        T=y.shape[0]

        [o,h]=self.forward_propagate(x)


        e = np.matmul(x, self.E.T)       # dim (T,d)
        assert e.shape == (T, self.emb_dim_size)

        for t in range(T)[::-1]:

            delta_1_t=(o[t]-y[t])     #dim (v)
            assert delta_1_t.shape[0]==(self.vocab_size)

            dldu+=np.outer(delta_1_t,h[t])                            #dim (v,dh)

            delta_2_t=np.matmul(delta_1_t,self.U)*h[t]*(1-h[t])       #dim(dh,)
            assert delta_2_t.shape[0]==self.hidden_size

            for bptt_step in range(0,T)[::-1]:
                e_t=e[bptt_step]                                      #dim(d,)
                assert e_t.shape[0]==(self.emb_dim_size)

                dldwe+=np.outer(delta_2_t,e_t)
                dldwh+=np.outer(delta_2_t,h[bptt_step-1])
                dldE.T[np.argmax(x[bptt_step])]+= (np.matmul(delta_2_t,self.We))

                delta_2_t=np.matmul(delta_2_t,self.Wh.T)*(h[bptt_step-1])*(1-h[bptt_step-1])
                assert delta_2_t.shape[0]==self.hidden_size

        assert dldu.shape==(self.vocab_size,self.hidden_size)
        assert dldwe.shape==(self.hidden_size,self.emb_dim_size)
        assert dldwh.shape==(self.hidden_size,self.hidden_size)
        return dldu,dldwe,dldwh,dldE


    def sgd_single_step(self,x,y,T,learning_rate=.001):
        """
        Run sgd for a given input x (hot encoded) and y after calculating backpropagation
        :param x: one hot encoded input sentence
        :param y: one hot encoded actual sentence
        """

        dldu, dldwe, dldwh,dldE=self.backpropogation(x,y)
        self.U-=learning_rate*dldu
        self.Wh-=learning_rate*dldwh
        self.We-=learning_rate*dldwe
        self.E-=learning_rate*dldE

    def calculate_total_loss(self,x_training,y_training,seq_len=10):

        """
        calculate loss on training data
        :param x_training,y_training: training data
        return Loss on whole trainging data
        """

        L=0
        for i in (range(len(x_training))):
            x_one_hot_encoded, y_one_hot_encoded = create_one_hot_encoding_for_sen \
                (x_training[i], y_training[i], self.vocab_to_int, self.int_to_vocab,seq_len)

            o,h=self.forward_propagate(x_one_hot_encoded)
            L+=-1* np.sum(np.log(np.sum(o*y_one_hot_encoded,axis=1)))
        return L

    def calculate_avg_loss(self,x_training,y_training,seq_len):

        N=len(x_training)
        return self.calculate_total_loss(x_training,y_training,seq_len)/N

    def calculate_loss_for_one_input(self,x,y):
        o, h = self.forward_propagate(x)
        loss=-1*np.sum(np.log(np.sum(o*y,axis=1)))
        return loss



    def train(self,x_training,y_training,seq_len=3,epochs=5,learning_rate=0.001):
        for i in range(epochs):
            loss=self.calculate_avg_loss(x_training,y_training,seq_len)

            print ("loss after {} epoch is {}".format(i,loss))

            for i in (range(len(x_training))):
                x_one_hot_encoded, y_one_hot_encoded = create_one_hot_encoding_for_sen \
                    (x_training[i], y_training[i], self.vocab_to_int, self.int_to_vocab,seq_len)

                self.sgd_single_step(x_one_hot_encoded,y_one_hot_encoded,seq_len,learning_rate)


    def gradient_check(self,x,y,h=.001):
        dldu, dldwe, dldwh, dldE=self.backpropogation(x,y)
        U=np.random.randn(self.vocab_size,self.hidden_size)
        #checking gradient for U
        U+=h
        loss_u_plush=self.calculate_loss_for_one_input(x,y)

        self.U-=2*h
        loss_u_minush = self.calculate_loss_for_one_input(x, y)

        return math.fabs((loss_u_plush-loss_u_minush)/2*h)-math.fabs(dldu)

#sample.txt is text file
input_file="sample.txt"


vocab_to_int,int_to_vocab=create_vocab_for_input_text(input_file)

x_training,y_training=create_training_examples_from_corpus(input_file)

x_one_hot_encoded, y_one_hot_encoded = create_one_hot_encoding_for_sen \
                    (x_training[0], y_training[0], vocab_to_int, int_to_vocab,10)


rnn=Rnn(vocab_to_int,int_to_vocab)
print (rnn.train(x_one_hot_encoded,y_one_hot_encoded))

