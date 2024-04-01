import pandas as pd
import numpy as np
import os
import csv
import json
import datetime
import matplotlib.pyplot as plt

class MatrixRecSysModel(object):
    def __init__(self,lr,
            batch_size,
            reg_p, 
            reg_q, 
            hidden_size=10, 
            epoch=10, 
            columns=["uid", "iid", "rating"],
            metric=None,):
        self.lr = lr # 学习率
        self.batch_size = batch_size
        self.reg_p = reg_p    # P矩阵正则系数
        self.reg_q = reg_q    # Q矩阵正则系数
        self.hidden_size = hidden_size  # 隐向量维度
        self.epoch = epoch    # 最大迭代次数
        self.columns = columns # 数据集的列名
        self.metric = metric # 评估指标
        self.loss_history = [] # 记录每一步的损失
        self.metric_history = []# 记录每一步的评估指标

    def loss(self,ground_truth, prediction):
        truth = np.array(ground_truth)
        pred = np.array(prediction)
        tmp = truth-pred
        loss = tmp.dot(tmp.T)
        return loss

    def loss_norm(self,ground_truth, prediction, P, Q, train_step_dataset):
        truth = np.array(ground_truth)
        pred = np.array(prediction)
        tmp = truth-pred
        loss = tmp.dot(tmp.T)
        ## 正则化项
        bias = 0
        for uid, iid, real_rating in train_step_dataset.itertuples(index=False):
            if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
                bias += 0
                continue
            p_u = P[uid]
            q_i = Q[iid]
            bias += np.dot(p_u,p_u.T)
            bias += np.dot(q_i,q_i.T)
        loss += (self.reg_p+self.reg_q)/2*bias
        return loss

    def grad_norm(self, P, Q, train_step_dataset):
        P_tmp = P
        Q_tmp = Q
        for uid, iid, real_rating in train_step_dataset.itertuples(index=False):
            if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
                continue
            p_u = P[uid]
            q_i = Q[iid]
            p_u_tmp = P_tmp[uid]
            q_i_tmp = Q_tmp[iid]
            r_ui = np.dot(p_u,q_i.T)
            dp_u = self.reg_p*p_u-2*(real_rating-r_ui)*q_i
            dq_i = self.reg_q*q_i-2*(real_rating-r_ui)*p_u
        # return dp_u,dq_i
            P_tmp[uid] = p_u_tmp-self.reg_p*dp_u
            Q_tmp[iid] = q_i_tmp-self.reg_q*dq_i
        P = P_tmp
        Q = Q_tmp
        return P, Q


    def load_dataset(self,train_dataset,dev_dataset):
        self.train_dataset = pd.DataFrame(train_dataset)
        self.dev_dataset = pd.DataFrame(dev_dataset)
        # for i in range (10):
        #     print("iter_test:")
        #     print(self.train_dataset[i*10:(i+1)*10])

        self.users_ratings = train_dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = train_dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        ## 一位用户其评价了的电影以及对应的评分
        ## 一场电影评价了它的所有用户以及对应的所有评分

        ## 所有评分的均值
        self.globalMean = self.train_dataset[self.columns[2]].mean()

    def init_matrix(self):
        
        P = dict(zip(
            self.users_ratings.index,
            np.random.randn(len(self.users_ratings), self.hidden_size).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.randn(len(self.items_ratings), self.hidden_size).astype(np.float32)
        ))
        return P, Q

    def train(self,optimizer_type: str):
        starttime = datetime.datetime.now()
        P,Q = self.init_matrix()
        best_metric_result = None
        best_P, best_Q = P, Q
        print("touch train")

        for i in range(self.epoch):
            print("Epoch time : ",i)
            if optimizer_type == "SGD":
                P,Q = self.sgd(P,Q)
            elif optimizer_type == "BGD":
                print("BGD")
                P,Q = self.bgd(P,Q,self.batch_size)
            else:
                raise NotImplementedError("Please choose one of SGD and BGD.")
            
            print("epoch:")
            print("epoch: {}/{} ,loss: {}".format(i,self.epoch,self.loss_history[-1]))
            metric_result = self.eval(P,Q)
            print("Current dev metric result: {}".format(metric_result))
            self.metric_history.append(metric_result)
            # print("Current dev loss result: {}".format(loss_test))
            if best_metric_result is None or metric_result <= best_metric_result:
                best_metric_result = metric_result
                best_P, best_Q = P, Q
                print("Best dev metric result: {}".format(best_metric_result))
        endtime = datetime.datetime.now()
        print("training durtion: ",(endtime-starttime).seconds)
        x_value = [i for i in range(self.epoch)]
        y_value = self.metric_history
        plt.plot(x_value,y_value)
        plt.show()
        
        np.savez("best_pq.npz", P=best_P, Q=best_Q)
        return 

    def sgd(self, P, Q):
        '''
        *********************************
        基本分：请实现【批量梯度下降】优化
        加分项：进一步优化如下
        - 考虑偏置项
        - 考虑正则化
        - 考虑协同过滤
        *********************************
        '''
        num_train = len(self.train_dataset)
        # mask = np.random.randint(0,num_train-1)
        # # print("mask",mask)
        # # print("self.train_dataset[0]",self.train_dataset[mask:mask+1])
        # # exit(0)
        # train_step_dataset = self.train_dataset[mask:mask+1]
        train_step_dataset = self.train_dataset.sample(1)
        # print("self.train_dataset\n",train_step_dataset)
        # exit(0)
        train_loss = 0.
        prediction, ground_truth = list(), list()
        for uid, iid, real_rating in train_step_dataset.itertuples(index=False):
            prediction_rating = self.predict_user_item_rating(uid, iid, P, Q)
                # dev_loss += abs(prediction_rating - real_rating)
            prediction.append(prediction_rating)
            ground_truth.append(real_rating)
        train_loss = self.loss_norm(ground_truth,prediction,P,Q,train_step_dataset)
        # print("batch_size: {}/{}, loss : {}".format((i+1)*batch_size, num_train, train_loss))
        # print("batch_size: {}/{}".format((i+1)*batch_size, num_train))
        self.loss_history.append(train_loss)
        ## 利用batch_size样本进行梯度更新
        P,Q = self.grad_norm(P, Q, train_step_dataset)
        return P, Q

    def bgd(self, P, Q, batch_size):
        '''
        *********************************
        基本分：请实现【批量梯度下降】优化
        加分项：进一步优化如下
        - 考虑偏置项
        - 考虑正则化
        - 考虑协同过滤
        *********************************
        '''
        num_train = len(self.train_dataset)
        iterations_per_epoch = max(num_train // batch_size, 1)
        for i in range(iterations_per_epoch):
            ## 按照顺序取出batch_size个样本对P,Q进行优化
            # train_step_dataset = self.train_dataset[i*batch_size:(i+1)*batch_size]
            train_step_dataset = self.train_dataset.sample(batch_size,replace=False)
            # print("train_step_dataset: ",train_step_dataset)
            train_loss = 0.
            prediction, ground_truth = list(), list()
            for uid, iid, real_rating in train_step_dataset.itertuples(index=False):
                prediction_rating = self.predict_user_item_rating(uid, iid, P, Q)
                # dev_loss += abs(prediction_rating - real_rating)
                prediction.append(prediction_rating)
                ground_truth.append(real_rating)
            
            ## 计算这batch_size样本的损失
            # print("prediction: ",prediction)
            # print("ground_truth: ",ground_truth)

            train_loss = self.loss_norm(ground_truth,prediction,P,Q,train_step_dataset)
            print("batch_size: {}/{}, loss : {}".format((i+1)*batch_size, num_train, train_loss))
            # print("batch_size: {}/{}".format((i+1)*batch_size, num_train))
            self.loss_history.append(train_loss)

            ## 利用batch_size样本进行梯度更新
            P,Q = self.grad_norm(P, Q, train_step_dataset)
        # print(train_loss)
        return P, Q
    
    def predict_user_item_rating(self, uid, iid, P, Q):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = P[uid]
        q_i = Q[iid]

        return np.dot(p_u, q_i)
    
    def eval(self, P, Q):
        # 根据当前的P和Q，在dev上进行验证，挑选最好的P和Q向量
        dev_loss = 0.
        prediction, ground_truth = list(), list()
        for uid, iid, real_rating in self.dev_dataset.itertuples(index=False):
            prediction_rating = self.predict_user_item_rating(uid, iid, P, Q)
            # dev_loss += abs(prediction_rating - real_rating)
            prediction.append(prediction_rating)
            ground_truth.append(real_rating)
        
        # metric_result = self.loss(ground_truth, prediction)
        # dev_loss = self.loss_norm(ground_truth, prediction, P, Q, self.dev_dataset)
        metric_result = self.metric(ground_truth, prediction)
        # dev_loss = self.loss(ground_truth, prediction)
        return metric_result


    def test(self, test_data):
        test_data = pd.DataFrame(test_data)
        # print(test_data[:10])
        # exit(0)
        # 加载训练好的P和Q
        best_pq = np.load("best_pq.npz", allow_pickle=True)
        P, Q = best_pq["P"][()], best_pq["Q"][()]

        save_results = list()
        for uid, iid in test_data.itertuples(index=False):
            pred_rating = self.predict_user_item_rating(uid, iid, P, Q)
            save_results.append(pred_rating)
        
        log_path = "results.csv"
        if os.path.exists(log_path):
            os.remove(log_path)
        file = open(log_path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'ID', 'rating'])
        for ei, rating in enumerate(save_results):
            csv_writer.writerow([ei, rating])
        file.close()





def mbgd(self, P, Q, batch_size):

    train_step_dataset = self.train_dataset.sample(batch_size)
        # print("train_step_dataset: ",train_step_dataset)
train_loss = 0.
prediction, ground_truth = list(), list()
for uid, iid, real_rating in train_step_dataset.itertuples(index=False):
    prediction_rating = self.predict_user_item_rating(uid, iid, P, Q)
    # dev_loss += abs(prediction_rating - real_rating)
    prediction.append(prediction_rating)
    ground_truth.append(real_rating)

## 计算这batch_size样本的损失
# print("prediction: ",prediction)
# print("ground_truth: ",ground_truth)

train_loss = self.loss_norm(ground_truth,prediction,P,Q,train_step_dataset)
# print("batch_size: {}/{}, loss : {}".format((i+1)*batch_size, num_train,train_loss))
# print("batch_size: {}/{}".format((i+1)*batch_size, num_train))
self.loss_history.append(train_loss)
## 利⽤batch_size样本进⾏梯度更新
P,Q = self.grad_norm(P, Q, train_step_dataset)
# print(train_loss)
return P, Q