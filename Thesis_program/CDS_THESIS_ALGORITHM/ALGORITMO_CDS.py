# -*- coding: utf-8 -*-
"""

THESIS ALGORITHM FOR X-RAY DATASET ANALYSIS 
''''''''''''''''''''''''''''''''''''''''''''''''''''''
AUTHOR: DAVID DRUMOND 
DATA: YEAR 
LOCAL: UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL 

FLUXOGRAMA DE AUTOMATIZAÇÃO PARA RECONHECIMENTO DE VARIÁVEIS GEOMETALÚRGICAS 

-> OBTENÇÃO DE DADOS
-> TRANSFORMAÇÃO EM INDICADORES 
-> ANÁLISE EXPLORATÓRIA DOS DADOS
-> CLASSIFICAÇÃO 
-> VALIDAÇÃO DOS MODELOS 
-> SALVAR MELHOR MODELO DE CLASSIFICAÇÃO 

"""


from __future__ import unicode_literals
import numpy as np 
import pandas as pd  
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.preprocessing as prepro
from subprocess import call
from sklearn.tree import export_graphviz
from sklearn.preprocessing import quantile_transform
from FUNCOES import *
from sklearn import preprocessing
import warnings
import time

# I don`t care about warnings
warnings.filterwarnings('ignore')



def main(planilha_saida, count_importance, grupos_name, numero_grupos, 
         ndesvpad,nome_sheet_dados, nome_sheet_variaveis, 
         cut_off_ouro, cut_off_enxofre,minerio_esteril= "False"):
    
    #..................................................................................................
    # GET DATA 
    dados_planilha, variaveis = ReadData(endereco_dados,nome_sheet_dados, nome_sheet_variaveis)
    writer = pd.ExcelWriter(planilha_saida)
    #..................................................................................................
    
    print("Calculando transformação de variáveis" )
    print("..............................................................................................")
    if minerio_esteril == True:
        
        #....................................................................................................
        # TRANSFORM DATA IN LABEL INDICADORS (ORE AND WASTE) ACCORDING MINERALOGY AND GRADE    
        dados_planilha['minerio_esteril'] = np.where(np.logical_or(dados_planilha['MINERALOGIA'] == "MINERIO", dados_planilha['Au [ppm]'] >=cut_off_ouro)  , 'MINERIO', 'ESTERIL')
        del variaveis[0]
        variaveis.insert(0,'minerio_esteril') 
        filter_dados_planilha = dados_planilha[variaveis].dropna()
        labels_input = variaveis[1:]
        Xdata = np.array(filter_dados_planilha[variaveis[1:]] )
        Ydata = np.array(filter_dados_planilha[variaveis[0]])
        Xnormalizer = Xdata
        #...................................................................................................
        
        # PREPROCESSING DATA 
        #...................................................................................................
        #Xnormalizer, Ydata = Remover_outliers (Ydata, Xdata, writer, ndesvpad)
        #Xnormalizer = prepro.PolynomialFeatures(degree=3, interaction_only=True).fit_transform(Xnormalizer)     
        #Xnormalizer = prepro.FunctionTransformer(np.log1p, validate=True).transform(Xnormalizer)
        #Xnormalizer = prepro.Normalizer().fit(Xnormalizer).transform(Xnormalizer)
        #Xnormalizer = prepro.FunctionTransformer(np.sqrt, validate=True).transform(Xnormalizer)
        #Xnormalizer = prepro.quantile_transform(Xnormalizer, output_distribution = 'normal')
        #....................................................................................................
        
    else:
        # RETIRAR VALORES NAN 
        filter_dados_planilha = dados_planilha[variaveis].dropna() 
        # ESTABELECER LABELS DOS VALORES 
        labels = dados_planilha['CLASSE MINERAL']        
        labels_input = variaveis[1:]
        # ESTABELECER VALORES DE ENTRADA E SAIDA 
        Xdata = np.array(filter_dados_planilha[variaveis[1:]] )
        Ydata = np.array(filter_dados_planilha[variaveis[0]])
        # HCA (Xdata, Ydata)
        Xnormalizer, Ydata = Remover_outliers (Ydata, Xdata, writer, ndesvpad) 
    print("Terminando cálculo de variaveis" )
    print(".............................................................................................")

    
    print("Análise exploratória de dados")
    print(".............................................................................................")
    #....................................................................................................
    # EXPLORATORY DATA ANALYSIS 
    print_contagem(Ydata)
    Importancia_variavies (variaveis, Xnormalizer, Ydata, writer)
    correlation_matrix(filter_dados_planilha[variaveis[1:]], variaveis[1:])
    histogram (filter_dados_planilha , variaveis[1:], len(variaveis))
    MDS(filter_dados_planilha[variaveis[1:]], variaveis[1:])
    accumulative_importance = PCA_importancia (labels_input, Xnormalizer, writer)
    #....................................................................................................
    print(".............................................................................................")
    
    
    #....................................................................................................
    # USE PCA FOR CLASSIFICATION
    X_PCA = Make_PCA_graph(3, Xnormalizer, Ydata) 
    #....................................................................................................
    
    print("Análise exploratória de dados")
    print(".............................................................................................")
    # CLASSIFICATION MODELS
    # sort only admissible models ["LOGISTIC", "LDA", "SVM", "RF", "NEURAL", "KNC", "DT", "NB"]
    list_of_labels  = ["LOGISTIC", "LDA", "SVM", "RF", "NEURAL", "KNC", "DT", "NB"]
    list_of_classificators = Classificators(X_PCA, Ydata, list_of_labels)
    classificators = [list_of_classificators[i] for i in list_of_labels] 
    #....................................................................................................
    print(".............................................................................................")
    
    print("MÉTRICAS DE CLASSIFICAÇÃO")
    print(".............................................................................................")
    #....................................................................................................
    #CLASSIFICATION METRICS
    #Create_ROC_curve(X_PCA, Ydata,list_of_classificators['NEURAL'])
    Prever_R_met(list_of_classificators['NEURAL'] , dados_planilha, X_PCA)
    Create_metrics_relatory(classificators,list_of_labels,X_PCA, Ydata)
    Create_classification_charts(X_PCA, Ydata, classificators, list_of_labels, minerio_esteril, caprend = True, cconf = True)
    #....................................................................................................
    print(".............................................................................................")
    
    
    #time.sleep(14000)
        
    

if __name__=="__main__":
    
    # INPUTS DO PROGRAMA 
    
    print("THESIS ALGORITHM FOR X-RAY DATASET ANALYSIS")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("AUTHOR: DAVID DRUMOND")
    print("DATA: 2019") 
    print("LOCAL: UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL") 
    
    print("FLUXOGRAMA DE AUTOMATIZAÇÃO PARA RECONHECIMENTO DE VARIÁVEIS GEOMETALÚRGICAS") 
    
    print("-> OBTENÇÃO DE DADOS")
    print("-> TRANSFORMAÇÃO EM INDICADORES") 
    print("-> ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("-> CLASSIFICAÇÃO") 
    print("-> VALIDAÇÃO DOS MODELOS") 
    print("-> SALVAR MELHOR MODELO DE CLASSIFICAÇÃO") 
    print (".......................................................")
    
  
    
    
    endereco_dados = 'BANCO DE DADOS.xlsx'
    nome_sheet_dados = 'TODOS  20-45 mm '
    nome_sheet_variaveis = 'Variaveis'
    planilha_saida = 'relatorio.xlsx'
    grupos_name = 'grupos.xlsx'
    count_importance = 0.95
    numero_grupos = 3
    ndesvpad = 5
    minerio_esteril = True
    cut_off_ouro = 0.45
    cut_off_enxofre = 0.2
    
    main(planilha_saida, count_importance, grupos_name, numero_grupos, 
         ndesvpad,nome_sheet_dados, nome_sheet_variaveis, 
         cut_off_ouro, cut_off_enxofre, minerio_esteril)



 