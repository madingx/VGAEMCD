import os
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
import keras.backend as K
from scipy.stats import chi2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class OCC_training_module_and_gene_prediction_module(object):

    def __init__(self, sess, n_nodes,train,test1,test0 ,args):
        self.sess = sess
        self.trains = tf.compat.v1.placeholder(tf.float32, shape=train.shape, name='train')#the embeddings of training positive examples
        self.test1=tf.compat.v1.placeholder(tf.float32, shape=test1.shape, name='test1')#the embeddings of test positive examples
        self.test0=tf.compat.v1.placeholder(tf.float32, shape=test0.shape, name='test0')#the embeddings of unlabeled examples
        self.learning_rate = args.learning_rate
        self.max_iteration = args.max_iteration
        self.n_nodes=n_nodes
        self.shape = np.array([self.n_nodes, self.n_nodes])
        self.positive_index=K.cast([i for i in range(self.trains.shape[0])],dtype='int64')
        self.predict_positive_index=K.cast([i+self.trains.shape[0] for i in range(self.test1.shape[0])],dtype='int64')
        self.predict_unlabeled_index=K.cast([i+self.trains.shape[0]+self.test1.shape[0] for i in range(self.test0.shape[0])],dtype='int64')
        self.chi_square_distance_threshold = chi2.isf(q=0.95, df=train.shape[0] - (train.shape[1] +train.shape[0]+1)/2)
        self._build_OCC_training_module_and_gene_prediction_module()

    def _build_OCC_training_module_and_gene_prediction_module(self):
        '''
        Here,for each iteration,we input training example results of fcn into loss function for our training model.
        Simultaneously ,we input unlabeled examples and test positive examples into the model for predicting gene label.
        :return:
        the prediction result of all condidate genes(all genes)
        '''

        print('....................................Training Stage................................................')

        z_encoded,x1,x2 = self.fcn()#fcn layers
        self.test1_sample=x1
        self.test0_sample=x2
        self.emb = z_encoded
        self.all_sample = K.concatenate((z_encoded,x1, x2), axis=0)
        self.location_layer=tf.reduce_mean(self.emb, axis=0, keepdims=True)#location layer
        self.scatter_layer=tf.matmul(tf.transpose(self.emb-self.location_layer),
                                     self.emb-self.location_layer)/tf.cast(tf.shape(self.emb)[0]-1, tf.float32)#scatter_layer
        self.inv_conv=tf.compat.v1.matrix_inverse(self.scatter_layer)
        total_maha_distance = tf.matmul(tf.matmul(self.emb - self.location_layer, self.inv_conv), tf.transpose(self.emb -
                                                                                                               self.location_layer))
        loss = K.sum(K.relu(tf.compat.v1.diag_part(total_maha_distance) -
                            self.chi_square_distance_threshold)) / self.emb.shape[0]#chi_square based loss function
        self.loss = loss
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print('....................................Test Stage................................................')
        total_maha_distance3 = tf.matmul(tf.matmul(self.all_sample - self.location_layer, self.inv_conv),
                                         tf.transpose(self.all_sample - self.location_layer))
        cc = (tf.compat.v1.diag_part(total_maha_distance3) <= self.chi_square_distance_threshold)
        self.prediction_result=tf.cast(cc, dtype=tf.int32)#0:negative examples,1:positive examples
        self.each_example_score = tf.compat.v1.diag_part(total_maha_distance3)#the scores of all genes



    def fcn(self):
        each_layer_dimention=[32,32,16]#the output dimension of each layer in fcn
        x1=self.trains# embeddings of training positive examples
        x2=self.test1# embeddings of test positive examples
        x3=self.test0#embeddings of unlabeled examples
        x = K.concatenate((x1, x2,x3), axis=0)
        x = Flatten(name='Flatten1')(x)
        print(x.shape)
        x = Dense(units=x.shape[1], name='embedding1')(x)
        print(x.shape)
        x = Dense(units=each_layer_dimention[0], name='embedding26')(x)
        print(x.shape)
        x = Flatten(name='Flatten1')(x)
        print(x.shape)
        x = Dense(units=each_layer_dimention[1], name='embedding23')(x)
        print(x.shape)
        x = Dense(units=each_layer_dimention[2], name='embedding21')(x)
        output_of_training_example = K.gather(x, self.positive_index)
        output_of_test_example=K.gather(x, self.predict_positive_index)
        output_of_unlabeled_example=K.gather(x, self.predict_unlabeled_index)
        return output_of_training_example,output_of_test_example,output_of_unlabeled_example



    def train(self, args, X_train, X_test,X123):
        train=np.array(X_train)
        test1=np.array(X_test)
        test0=np.array(X123)
        feed_dict = {self.trains:train,self.test1:test1,self.test0:test0}
        for i in range(self.max_iteration):
            print('wsk', i)
            _, loss,prediction_result,scores= self.sess.run([self.train_step,
                                                                                                      self.loss
                                                                                                ,self.prediction_result
                                                            ,self.each_example_score


                                                                                                  ], \
 \
                                                                                                     feed_dict=feed_dict)

            print("At step {0} \n Loss: {1} \n.".format(i, loss))
            print('prediction result', prediction_result)
            print('scores', scores)





