# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:08:43 2018

@author: lankuohsing
"""
# In[]
import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector
# In[]

class LstmRNN(object):
    def __init__(self, sess,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=21,
                 output_size=1,
                 logs_dir="logs",
                 plots_dir="figures",
                 max_epoch=5):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.output_size=output_size
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.max_epoch=max_epoch
        self.build_graph()


    def build_graph(self):
        """
        The model asks for 4 things to be trained:
        - learning_rate
        - keep_prob: 1 - dropout rate
        - input: training data X
        - targets: training label y
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.num_steps,self.output_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()



        print( "inputs.shape:", self.inputs.shape)


        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32, scope="dynamic_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        #val = tf.transpose(val, [1, 0, 2])
        val=tf.reshape(val,[-1,self.lstm_size])
        print("val.shape:",val.shape)
        #last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")#取出最后一个输出值,也即第num_steps-1个值

        weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.output_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="b")
        #last.get_shape()=[batch_size,lstm_size]
        #pred.get_shape()=[batch,input_size]
        self.pred = tf.matmul(val, weights) + bias#pred.get_shape=[output_size]
        print("pred.shape:",self.pred.shape)
        print("targets.shape:",self.targets.shape)
        '''
        为tensorboard准备数据
        '''
        self.last_sum = tf.summary.histogram("lstm_state", val)
        self.w_sum = tf.summary.histogram("w", weights)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)
        self.pred=tf.reshape(self.pred,[-1])
        self.targets=tf.reshape(self.targets,[-1])
        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        #self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        a0=self.pred - self.targets
        a=tf.cast(tf.sign(a0)*a0,tf.float32)/(11.5-1.5*tf.cast(tf.sign(a0),tf.float32))
        b=tf.exp(tf.cast(a, tf.float32))-1
        self.loss=tf.reduce_sum(b)
        #self.loss = tf.reduce_sum(tf.cast(tf.exp(tf.abs(self.pred - self.targets)/10),tf.float32)-1, name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

        # Separated from train loss.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()
    def test(self, dataset_RUL, config):
        final_text_X_list=dataset_RUL.final_test_X_list
        final_text_X_list_indices=list(range(len(final_text_X_list)))
       #random.shuffle(test_list_indices)#随机打乱
       #sample_indices=test_list_indices[0:config.sample_size]
        sample_indices=final_text_X_list_indices
        test_pred_list=[]
        for indice in sample_indices:
            sample_X=final_text_X_list[indice]


            test_data_feed = {
                    self.learning_rate: 0.0,
                    self.keep_prob: 1.0,
                    self.inputs: sample_X

                    }
            test_pred = self.sess.run([self.pred], test_data_feed)
            test_pred_list.append(test_pred)
        return test_pred_list
    def train(self, dataset_RUL, config):
        """
        Args:
            dataset_RUL (dataset_RUL)
            config (tf.app.flags.FLAGS)
        """

        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)



        tf.global_variables_initializer().run()

        # In[]
        # Merged test data of different stocks.
        test_X_list = dataset_RUL.test_X_list
        test_y_list = dataset_RUL.test_y_list

        # In[]
        '''
        test_X_np=test_X_list[0]
        test_y_np=test_y_list[0]
        for i in range(1,len(test_X_list)):
            test_X_np=np.vstack((test_X_np,test_X_list[i]))
            test_y_np=np.vstack((test_y_np,test_y_list[i]))
        print( "len(test_X_np) =", len(test_X_np))
        print( "len(test_y_np) =", len(test_y_np))


        test_y_np_flattened=test_y_np.reshape((-1,))
        test_data_feed = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: test_X_np,
            self.targets: test_y_np_flattened,

        }
        '''
        global_step = 0
        #注：array也可以用len函数
        num_batches = len(dataset_RUL.train_X)// config.batch_size#
        random.seed(time.time())
        # In[]
        '''
        随机挑选一些发动机，绘制预测/真实寿命曲线
        '''
        test_list_indices=list(range(len(dataset_RUL.test_X_list)))
       #random.shuffle(test_list_indices)#随机打乱
       #sample_indices=test_list_indices[0:config.sample_size]
        sample_indices=test_list_indices
        # In[]
        print( "Start training for RULs:")
        for epoch in list(range(config.max_epoch)):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )#早期的epoch（默认为5）之内，不对学习率进行衰减


            for batch_X, batch_y in dataset_RUL._generate_one_epoch(config.batch_size):
                epoch_step += 1# max_epoch_step=num_batches(+1)
                global_step += 1# max_global_step=max_epoch_step*max_epoch
                batch_y=batch_y.reshape((-1,))
                train_data_feed = {
                        self.learning_rate: learning_rate,
                        self.keep_prob: config.keep_prob,
                        self.inputs: batch_X,
                        self.targets: batch_y,

                    }
                train_loss, _, train_merged_sum = self.sess.run(
                        [self.loss, self.optim, self.merged_sum], train_data_feed)
                self.writer.add_summary(train_merged_sum, global_step=global_step)
                #全局训练次数（global_step)大于200时，开始测试
                if np.mod(global_step,   200) == 1:

                    print( "global step:%d [epoch:%d] [learning rate: %.6f] train_loss:%.6f" % (
                            global_step, epoch, learning_rate, train_loss))
                    if global_step>=5000 and np.mod(global_step,1000)==1:
                        # Plot samples
                        for indice in sample_indices:
                            sample_X=test_X_list[indice]
                            sample_y=test_y_list[indice]
                            sample_y_flattened=sample_y.reshape((-1,))
                            test_data_feed = {
                                    self.learning_rate: 0.0,
                                    self.keep_prob: 1.0,
                                    self.inputs: sample_X,
                                    self.targets: sample_y_flattened,
                                    }
                            test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)
                            #image_path = os.path.join(self.model_plots_dir, "epoch{:02d}_step{:04d}_indice{:04d}.png".format(
                                     epoch, epoch_step,indice))
                            sample_pred = test_pred
                            sample_truth = sample_y_flattened
                            #self.plot_samples(sample_pred, sample_truth, image_path)

                    self.save(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save(global_step)
        return final_pred

    @property
    def model_name(self):
        name = "RUL_lstm%d_num_layers%d_numstep%d_input%d_maxepoch%d" % (
            self.lstm_size, self.num_layers, self.num_steps, self.input_size,self.max_epoch)


        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot_samples(self, sample_pred, sample_truth, image_path):
        figure=plt.figure()
        figure.set_figheight(5)
        figure.set_figwidth(8)
        plot_test, = plt.plot(sample_truth, label='real_RUL')
        plot_predicted, = plt.plot(sample_pred, label='predicted_RUL')
        plt.legend([plot_predicted, plot_test],['predicted', 'truth'])
        '''
        x_start=1000
        x_end=1060
        y_start=-1
        y_end=-0.2
        '''
        #plt.axis([x_start,x_end,y_start,y_end])

        plt.savefig(image_path+'.png')
        plt.close()
