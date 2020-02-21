#!/usr/bin/env python
# coding: utf-8

# In[12]:


import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy as sc
from scipy import stats
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score,recall_score, f1_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
import seaborn as sns
import numpy_indexed as npi

get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


class METRICS():
    
    def build_model(self,model, parameters, folds, train_size, X,Y,groups_original):
        acc_scorer = make_scorer(accuracy_score)
        recalls = make_scorer(recall_score,average='micro')##buscar por que micro
        precision = make_scorer(precision_score,average='micro')
        f1 = make_scorer(f1_score,average='micro')
        scores =  {'recalls':recalls,'precision':precision,'f1':f1,'Accuracy': make_scorer(accuracy_score)}
        gss = GroupShuffleSplit(n_splits=folds, train_size=train_size, random_state=0)
        model = GridSearchCV(model,parameters,scores,-1,refit='Accuracy',return_train_score=True, cv=gss.split(X, Y, groups=groups_original))
        model.fit(X,Y)
        return model 
    
    def metrics(self,model,X_test,Y_test):

      y_predicted = model.predict(X_test)
      print('Accuracy: ', accuracy_score(Y_test, y_predicted), '\n')
      report = classification_report(Y_test, y_predicted)
      print("\nclassification report :\n",report )

      # Matriz de confusión
      cm = confusion_matrix(Y_test, y_predicted)
      # Normalise
      cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      fig, ax = plt.subplots(figsize=(10,10))

      sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=clases, yticklabels=clases)
      plt.ylabel('Actual')
      plt.xlabel('Predicted')
      ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
      plt.show(block=False)

      #sns.heatmap(cm,annot=True,fmt = "d",linecolor="k",linewidths=3)
      #plt.title("Matriz de confusión",fontsize=20

      return report

    
    def Metrics(self,model,X_test,Y_test,tracks_test,clases):
        y_predicted = model.predict(X_test)

        X_test_with_predict_trak= np.column_stack((X_test,y_predicted))  

        X_test_with_id_trak = np.column_stack((X_test_with_predict_trak,tracks_test))

        index_traks = list(np.unique(tracks_test))

        prediction_by_instances = npi.group_by(X_test_with_id_trak[:, -1]).split(X_test_with_id_trak[:, -2])

        real_classifier = np.array_split(Y_test,len(index_traks))

        labels = []

        for j in range(len(real_classifier)):
            clase = stats.mode(real_classifier[j])[0][0][0]
            labels.append(clase)
        print('verdaderos',labels)



        predictions = []

        for v in range(prediction_by_instances.shape[0]):
            decision = stats.mode(prediction_by_instances[v])[0][0]
            predictions.append(decision)
        print('predi',predictions)


        print('Accuracy: ', accuracy_score(labels, predictions), '\n')
        report = classification_report(labels, predictions)
        print("\nclassification report :\n",report )

      # Matriz de confusión
        cm = confusion_matrix(labels, predictions)
      # Normalise
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))

        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=clases, yticklabels=clases)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
        plt.show(block=False)

      #sns.heatmap(cm,annot=True,fmt = "d",linecolor="k",linewidths=3)
      #plt.title("Matriz de confusión",fontsize=20)
      
        return report
    def learning_curve(self,model,best_parameters,folds, X,Y,groups, suptitle='', title='', xlabel='Training Set Size', ylabel='Acurracy'):
        """
        Parameters
        ----------
        suptitle : str
            Chart suptitle
        title: str
            Chart title
        xlabel: str
            Label for the X axis
        ylabel: str
            Label for the y axis
        Returns
        -------
        Plot of learning curves
        """

        # create lists to store train and validation scores
        train_score = []
        val_score = []
        std_train= []
        std_val=[]

        # create ten incremental training set sizes
        training_set_sizes = np.linspace(.1,.9,5).tolist()
        # for each one of those training set sizes

        for i in training_set_sizes:  
            model_trained = self.build_model(self=0,model=model, parameters=best_parameters, folds=folds, train_size=i, X=Y,Y=Y,groups_original = groups)                
            EfficiencyVal= model_trained.cv_results_['mean_test_Accuracy'][model_trained.best_index_]
            EfficiencyTrain=model_trained.cv_results_['mean_train_Accuracy'][model_trained.best_index_]
            stdTrain=model_trained.cv_results_['std_train_Accuracy'][model_trained.best_index_]
            stdVal=model_trained.cv_results_['std_test_Accuracy'][model_trained.best_index_]

            # store the scores in their respective lists
            train_score.append(EfficiencyTrain)
            val_score.append(EfficiencyVal)
            std_train.append(stdTrain)
            std_val.append(stdVal)

        train_score =np.array(train_score)
        val_score =np.array(val_score)
        std_train =np.array(std_train)
        std_val =np.array(std_val)


        # plot learning curves
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.plot(training_set_sizes, train_score, c='gold')
        ax.plot(training_set_sizes, val_score, c='steelblue')

        ax.fill_between(training_set_sizes,train_score+std_train,train_score-std_train,facecolor='gold',alpha=0.5)
        ax.fill_between(training_set_sizes,val_score+std_val,val_score-std_val,facecolor='steelblue',alpha=0.5)

        # format the chart to make it look nice
        fig.suptitle(suptitle, fontweight='bold', fontsize='20')
        ax.set_title(title, size=20)
        ax.set_xlabel(xlabel, size=16)
        ax.set_ylabel(ylabel, size=16)
        ax.legend(['Train set', 'Test set'], fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim(0, 1)

        def percentages(x, pos):
            """The two args are the value and tick position"""
            if x < 1:
                return '{:1.0f}'.format(x*100)
            return '{:1.0f}%'.format(x*100)

        def numbers(x, pos):
            """The two args are the value and tick position"""
            if x >= 1000:
                return '{:1,.0f}'.format(x)
            return '{:1.0f}'.format(x)
        data = {'Train_Size':training_set_sizes, 'mean_train_Accuracy':train_score,'mean_test_Accuracy':val_score,'std_train_Accuracy':std_train,'std_test_Accuracy':std_val}
        df_split_params = pd.DataFrame(data)
        return df_split_params


# In[ ]:




