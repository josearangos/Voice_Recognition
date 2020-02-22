#!/usr/bin/env python
# coding: utf-8

# In[10]:


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

class PIPELINE_MULTIPLES_INST:

#################################################################################################################
    def __init__(self,X,Y,groups,tracks,FOLDS,TRAIN_SIZE,clases):
        self.X=X
        self.Y=Y
        self.groups=groups
        self.tracks=tracks
        self.clases = clases
        self.folds=FOLDS
        self.train_size=TRAIN_SIZE
        
        returns = self.splitDataSet(X,Y,groups,tracks,0.2)       
        self.X_train_original=returns[0]
        self.Y_train__original=returns[1]
        self.X_test_original=returns[2]
        self.Y_test_original=returns[3]
        self.tracks_train_original=returns[4]
        self.tracks_test_original=returns[5]
        self.groups_original=returns[6]
        self.groups_test_original= returns[7]
        
        
        
        self.summaryInfo()

#################################################################################################################
    
    def summaryInfo(self):
        
        ### Número de clases
        n_classes=len(np.unique(self.Y))
        print('Número de clases:', n_classes)
        
        ### Número de hablantes
        n_groups = len(np.unique(self.groups))
        print('Número de hablantes diferentes', n_groups)
        
        ### Número de Audios
        n_tracks = len(np.unique(self.tracks))
        print('Número de audios diferentes',n_tracks)
        
        ## Dividir el dataset (train/test) 0.8/0.2
        
        print("### Info TRAIN")
        print('X',self.X_train_original.shape)
        print('Groups',len(np.unique(self.groups_original)))
        print('Tracks',len(np.unique(self.tracks_train_original)))
        
        
        print("### Info TEST")
        print('X',self.X_test_original.shape)
        print('Groups',len(np.unique(self.groups_test_original)))
        print('Tracks',len(np.unique(self.tracks_test_original)))
        
    def splitDataSet(self,X,Y,groups,tracks,test_size):              
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        gss.get_n_splits()
        returns = []
        for train_index, test_index in gss.split(X, Y, groups=groups):
            X_train_original, X_test_original = X[train_index], X[test_index]
            #print(X_train, X_test)
            Y_train__original, Y_test_original = Y[train_index], Y[test_index]
            #print(y_train, y_test)
            groups_original = groups[train_index]
            groups_test_original=groups[test_index]
            tracks_train_original = tracks[train_index]
            tracks_test_original = tracks[test_index]  
            
            returns.append(X_train_original)
            returns.append(Y_train__original)
            returns.append(X_test_original)
            returns.append(Y_test_original)
            returns.append(tracks_train_original)
            returns.append(tracks_test_original)
            returns.append(groups_original)
            returns.append(groups_test_original)    
              
        return returns
              
    def modelPredict(self,model,Xtest,tracks_test):
        

        Yest = model.predict(Xtest)    

        #Creamos un matriz con la primera fila las predicciones y la otra el track      
        X_test_with_id_trak = np.column_stack((Yest,tracks_test))

        #Agrupamos por audio
        prediction_by_instances = npi.group_by(X_test_with_id_trak[:, -1]).split(X_test_with_id_trak[:, -2])

        #Sacamos la moda de prediccion
        predictions = []

        for v in range(len(prediction_by_instances)):
            decision = stats.mode(prediction_by_instances[v])[0]
            predictions.append(decision)
    
        return np.array(predictions) 
#################################################################################################################
    def createGroups(self, Y,tracks):
        Y_train_tracks = np.column_stack((Y,tracks))
        groups_by_tracks = npi.group_by(Y_train_tracks[:, -1]).split(Y_train_tracks[:, -2])
        modas = []

        for v in range(len(groups_by_tracks)):
            moda = stats.mode(groups_by_tracks[v])[0]
            modas.append(moda)
        return np.array(modas)
#################################################################################################################
    """
    """
    
    def TRAIN(self, model,label_request=False):
        gss = GroupShuffleSplit(n_splits=self.folds, train_size=.7)
        EficienciaTrain = np.zeros(self.folds)
        EficienciaVal = np.zeros(self.folds)
        j = 0
        
        model_trained=model

        for train_idx, test_idx in gss.split(self.X_train_original, self.Y_train__original, self.groups_original):

            X_train_fold =self.X_train_original[train_idx]
            Y_train_fold=self.Y_train__original[train_idx]
            X_test_fold=self.X_train_original[test_idx]
            Y_test_fold=self.Y_train__original[test_idx]

            tracks_train_fold=self.tracks_train_original[train_idx]
            tracks_test_fold=self.tracks_train_original[test_idx]

            #Entrenamiento
            if(label_request):
                model_trained.fit(X_train_fold,Y_train_fold)
            else:
                model_trained.fit(X_train_fold)

             #Validación
            Ytrain_pred = self.modelPredict(model_trained,X_train_fold,tracks_train_fold)

            Ytest_pred = self.modelPredict(model_trained,X_test_fold,tracks_test_fold)


            #Metricas en entrenamiento

            #Hacer groupby por tracks
            Y_real_train = self.createGroups(Y_train_fold,tracks_train_fold)
            Y_real_test = self.createGroups(Y_test_fold,tracks_test_fold)

            """
            print('TRAIN')
            print('Predicted',Ytrain_pred )
            print('Real',Y_real_train)

            print('----------------')
            print('TEST')
            print('Predicted',Ytest_pred)
            print('Real',Y_real_test)

            """

            EficienciaTrain[j] = np.mean(Ytrain_pred.ravel() == Y_real_train.ravel())
            EficienciaVal[j] = np.mean(Ytest_pred.ravel() == Y_real_test.ravel())
            j += 1

        eficiencia_Train=(np.mean(EficienciaTrain))
        intervalo_Train=(np.std(EficienciaTrain))
        eficiencia_Test=np.mean(EficienciaVal)
        intervalo_Test=np.std(EficienciaVal)
        #print('Eficiencia durante el entrenamiento = ' + str(eficiencia_Train) + '+-' + str(intervalo_Train))
        #print('Eficiencia durante la validación = ' + str(eficiencia_Test) + '+-' + str(intervalo_Test)) 
        #Se retorna el modelo del ultimo fold
        return model_trained,eficiencia_Train,intervalo_Train,eficiencia_Test,intervalo_Test
#################################################################################################################

#################################################################################################################
    """
    """
    
    def TRAIN_2(self, model,X,Y,tracks,i,folds,groups):
        gss = GroupShuffleSplit(n_splits=folds, train_size=.7)
        EficienciaTrain = np.zeros(folds)
        EficienciaVal = np.zeros(folds)
        j = 0
        
        model_trained=model

        for train_idx, test_idx in gss.split(X, Y, groups):

            X_train_fold =X[train_idx]
            Y_train_fold=Y[train_idx]
            X_test_fold=X[test_idx]
            Y_test_fold=Y[test_idx]

            tracks_train_fold=tracks[train_idx]
            tracks_test_fold=tracks[test_idx]

            #Entrenamiento
            model_trained.fit(X_train_fold)
            
             #Validación
            Ytrain_pred = self.modelPredict(model_trained,X_train_fold,tracks_train_fold)

            Ytest_pred = self.modelPredict(model_trained,X_test_fold,tracks_test_fold)


            #Metricas en entrenamiento

            #Hacer groupby por tracks
            Y_real_train = self.createGroups(Y_train_fold,tracks_train_fold)
            Y_real_test = self.createGroups(Y_test_fold,tracks_test_fold)

            """
            print('TRAIN')
            print('Predicted',Ytrain_pred )
            print('Real',Y_real_train)

            print('----------------')
            print('TEST')
            print('Predicted',Ytest_pred)
            print('Real',Y_real_test)

            """

            EficienciaTrain[j] = np.mean(Ytrain_pred.ravel() == Y_real_train.ravel())
            EficienciaVal[j] = np.mean(Ytest_pred.ravel() == Y_real_test.ravel())
            j += 1

        eficiencia_Train=(np.mean(EficienciaTrain))
        intervalo_Train=(np.std(EficienciaTrain))
        eficiencia_Test=np.mean(EficienciaVal)
        intervalo_Test=np.std(EficienciaVal)
        #print('Eficiencia durante el entrenamiento = ' + str(eficiencia_Train) + '+-' + str(intervalo_Train))
        #print('Eficiencia durante la validación = ' + str(eficiencia_Test) + '+-' + str(intervalo_Test)) 
        #Se retorna el modelo del ultimo fold
        return model_trained,eficiencia_Train,intervalo_Train,eficiencia_Test,intervalo_Test
#################################################################################################################



    #Xtest,Ytest,groupsTest,tracksTest,clases
    def confusion_matrix_Metrics(self, model):
 

        Ytest_pred= self.modelPredict(model,self.X_test_original,self.tracks_test_original)


        Y_real_test = self.createGroups(self.Y_test_original,self.tracks_test_original)

        print('Accuracy: ', accuracy_score(Y_real_test, Ytest_pred), '\n')
        report = classification_report(Y_real_test, Ytest_pred)
        print("\nclassification report :\n",report )

        # Matriz de confusión
        cm = confusion_matrix(Y_real_test, Ytest_pred)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=self.clases, yticklabels=self.clases)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
        plt.show(block=False)
        return report
    
    
#################################################################################################################

#folds, X,Y,groups,tracks
    def learning_curve(self, model,suptitle='', title='', xlabel='Training Set Size', ylabel='Acurracy'):
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
                    
            
            model, eficiencia_Train,intervalo_Train,eficiencia_Test,intervalo_Test=self.TRAIN_2(model,self.X,self.Y,self.tracks,i,self.folds,self.groups)
            
            
            # store the scores in their respective lists
            train_score.append(eficiencia_Train)
            val_score.append(eficiencia_Test)
            std_train.append(intervalo_Train)
            std_val.append(intervalo_Test)

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

