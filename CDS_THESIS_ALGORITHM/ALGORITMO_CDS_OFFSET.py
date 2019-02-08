# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 12:39:48 2018

@author: David
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
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import quantile_transform

from FUNCOES import *


def main(planilha_saida, count_importance, grupos_name, numero_grupos, 
         ndesvpad,nome_sheet_dados, nome_sheet_variaveis, 
         cut_off_ouro, cut_off_enxofre,minerio_esteril= "False"):
    
    # OBTER DADOS DA PLANILHA 
    
    dados_planilha, variaveis = ReadData(endereco_dados,nome_sheet_dados, nome_sheet_variaveis)
    
  
     
    # REMOVE OUTLIERS 
    writer = pd.ExcelWriter(planilha_saida)
    
    
    if minerio_esteril == True:
        dados_planilha['minerio_esteril'] = np.where(np.logical_or(dados_planilha['MINERALOGIA'] == "MINERIO", dados_planilha['Au [ppm]'] >=cut_off_ouro)  , 'MINERIO', 'ESTERIL')
        #dados_planilha['minerio_esteril'] = np.where(dados_planilha['Au [ppm]'] >=cut_off_ouro  , 'MINERIO', 'ESTERIL')
        del variaveis[0]
        variaveis.insert(0,'minerio_esteril') 
        filter_dados_planilha = dados_planilha[variaveis].dropna()
        #filter_dados_planilha = filter_dados_planilha[(dados_planilha['High_70%'] < 0.2)&(dados_planilha['Low_70%'] < 1000000)  ]
        labels_input = variaveis[1:]
        Xdata = np.array(filter_dados_planilha[variaveis[1:]] )
        Ydata = np.array(filter_dados_planilha[variaveis[0]])
        
        labels = dados_planilha['CLASSIFICACAO']
        
        
        Xnormalizer, Ydata = Remover_outliers (Ydata, Xdata, writer, ndesvpad)
        #Xnormalizer = prepro.PolynomialFeatures(degree=3, interaction_only=True).fit_transform(Xnormalizer)     
        #Xnormalizer = prepro.FunctionTransformer(np.log1p, validate=True).transform(Xnormalizer)
        #Xnormalizer = prepro.Normalizer().fit(Xnormalizer).transform(Xnormalizer)
        #Xnormalizer = prepro.FunctionTransformer(np.sqrt, validate=True).transform(Xnormalizer)
        #Xnormalizer = prepro.quantile_transform(Xnormalizer, output_distribution = 'normal')
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

    
    # DETERMINAR IMPORTANCIA DAS VARIAVEIS 
    
    Importancia_variavies (variaveis, Xnormalizer, Ydata, writer)

    # DETERMINAR COMPONENTES PRINCIPAIS DOS DADOS E IMPORTANCIA DOS VALORES 
    
    accumulative_importance = PCA_importancia (labels_input, Xnormalizer, writer)
    print ("IMPORTANCIA PCA")
    print ("....................")
    print ([[round(n, 5) for n in accumulative_importance]])
    
    #HCA(Xnormalizer, variaveis)
    

    # DETERMINAR SIMILARIDADE ENTRE GRUPOS
    
    # plotting the correlation matrix
    #correlation_matrix(filter_dados_planilha[variaveis[1:]], variaveis[1:])
    
    #if minerio_esteril == True:
    #    Curva_concentracao (filter_dados_planilha, variaveis, 0.0, 1, 50)

    #histogram (filter_dados_planilha[variaveis[1:]] , variaveis[1:])
    #print_contagem(Ydata)



    
    # REALIZAR TESTES DE AGRUPAMENTO
    LOGISTIC = LogisticRegression ()
    LDA = LinearDiscriminantAnalysis ()
    SVM = svm.LinearSVC()
    RF = RandomForestClassifier(max_depth =4, criterion ='entropy', min_samples_split= 15, random_state =0)
    DT = tree.DecisionTreeClassifier(max_depth =4, criterion ='entropy', min_samples_split= 15)
    NEURAL = MLPClassifier( solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 40), random_state=0)
    KNC = KNeighborsClassifier(n_neighbors= 30)
    NB = BernoulliNB(alpha = 1.0)

    
    #transform pca values
    pca = decomposition.PCA(n_components = 8)
    X_PCA2 = pca.fit_transform(Xnormalizer)
    X_PCA = pca.fit_transform(Xnormalizer)
    #X_PCA = Xnormalizer
    
    print_dados(Xnormalizer, Ydata,labels_input,labels)
    
    
    #make graph of pca values 
    df = pd.DataFrame(np.array([X_PCA2[:,0], X_PCA2[:,1], Ydata]).T, columns=["PCA1","PCA2", "Grupos"])
    sns.pairplot(x_vars=["PCA1"], y_vars=["PCA2"], data=df, hue='Grupos',markers ='+', size=7, plot_kws={'alpha': 0.8}  )
    plt.title("PCA Factors")
    plt.show()
    
    
    
    '''
    #improve TSNE algorithm
    tsne = TSNE(n_components = 2)
    X_PCA3 = tsne.fit_transform(Xnormalizer)
    df = pd.DataFrame(np.array([X_PCA3[:,0], X_PCA3[:,1], Ydata]).T, columns=["TSNE1","TSNE2", "Grupos"])
    sns.pairplot(x_vars=["TSNE1"], y_vars=["TSNE2"], data=df, hue='Grupos', size=5 )
    plt.title("TSNE Factors")
    plt.show()
    
    
    
    #X_PCA =  umap.UMAP().fit_transform(Xnormalizer)
    reducer = umap.UMAP().fit_transform(Xnormalizer)
    df = pd.DataFrame(np.array([reducer[:,0], reducer[:,1], Ydata]).T, columns=["UMAP1","UMAP2","Grupos"])
    um  = df["UMAP1"].astype(float)
    dois = df["UMAP2"].astype(float)
    print um.corr(dois)
    sns.pairplot(x_vars=["UMAP1"], y_vars=["UMAP2"], data=df, hue="Grupos",size=5)
    plt.title("UMAP")
    plt.show()
    '''
    
    LOGISTIC.fit(X_PCA, Ydata)
    LDA.fit(X_PCA, Ydata)
    SVM.fit(X_PCA, Ydata)
    DT.fit(X_PCA, Ydata)
    RF.fit(X_PCA, Ydata)
    NEURAL.fit(X_PCA, Ydata)
    KNC.fit(X_PCA, Ydata)
    NB.fit(X_PCA, Ydata)
    
    
    
    
    '''
    # DETERMINAÇÃO DA RECUPERAÇÃO METALURGICA
    predicted = NEURAL.predict(X_PCA)
    ouro = dados_planilha['Au [ppm]']
    peso = dados_planilha['Weight']
    
    
    plt.show()
    ouro_rejeito =[]
    ouro_alimentacao =[]
    ouro_concentrado = []
    massa_rejeito = 0
    massa_concentrado= 0
    massa_alimentacao = 0
    
   
    for j,i in enumerate(predicted):
        if i == 'MINERIO':
            if ouro[j] >= 0  and peso[j] > 0:
                ouro_concentrado.append(ouro[j]*peso[j])
                massa_concentrado += peso[j]
        if i == 'ESTERIL':
            if ouro[j] >= 0 and peso[j] > 0:
                ouro_rejeito.append(ouro[j]*peso[j])
                massa_rejeito += peso[j]
        if ouro[j] >= 0 and peso[j]>0:
            ouro_alimentacao.append(ouro[j]*peso[j])
            massa_alimentacao += peso[j]
                
    
    ouro_rejeito = np.array(ouro_rejeito)
    ouro_rejeito = ouro_rejeito[ouro_rejeito != None]

    ouro_concentrado = np.array(ouro_concentrado)
    ouro_concentrado= ouro_concentrado[ouro_concentrado!= None]

    ouro_alimentacao = np.array(ouro_alimentacao)
    ouro_alimentacao= ouro_alimentacao[ouro_alimentacao!= None]
    
    teor_concentrado = np.sum(ouro_concentrado)/massa_concentrado
    teor_alimentacao = np.sum(ouro_alimentacao)/massa_alimentacao
    print "Recuperação metalúrgica do ouro no ore sorting"
    print ".............................................."
    print teor_concentrado*massa_concentrado/(teor_alimentacao*massa_alimentacao)
    '''
    
    
    
    '''
    # Export as dot file
    dot_data = export_graphviz(DT, out_file='tree.dot', 
                feature_names = variaveis[1:],
                class_names = ['MINERIO', 'ESTERIL'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    graph = graphviz.Source(dot_data)
    graph
    # remember dot -Tpng tree.dot -o tree.png
    '''
    
    '''
    arquivo_saida =[]
    arquivo = open("relatorio_classificacao.txt","w")
    arquivo.write('modelo accuracy f-score precision recall area_aboce_roc \n' )
    from sklearn.model_selection import StratifiedKFold
    import sklearn.metrics as m
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    for modelo, nome in zip([LOGISTIC, LDA, SVM, RF, NEURAL, KNC, DT, NB], [" LOGISTIC", " LDA", " SVM", " RF", " NEURAL", " KNC", "DT", "NB"]):
        resultados_a = []
        resultados_f = []
        resultados_s = []
        resultados_r = []
        resultados_roc = []
        Ydata2 = np.array([1 if x == "MINERIO" else 0 for x in Ydata])
        for indice_treinamento, indice_teste in kfold.split(X_PCA,Ydata2):
            modelo.fit( X_PCA[indice_treinamento], Ydata2[indice_treinamento])
            previsoes = modelo.predict(X_PCA[indice_teste])
            resultados_a.append(m.accuracy_score(Ydata2[indice_teste], previsoes))
            resultados_f.append(m.f1_score(Ydata2[indice_teste], previsoes))
            resultados_s.append(m.precision_score(Ydata2[indice_teste], previsoes))
            resultados_r.append(m.recall_score(Ydata2[indice_teste], previsoes))
            resultados_roc.append(m.roc_auc_score(Ydata2[indice_teste], previsoes))
        resultados = np.asarray(resultados_a)
        resultados1 = np.asarray(resultados_f)
        resultados2 = np.asarray(resultados_s)
        resultados3 = np.asarray(resultados_r)
        resultados4 = np.asarray(resultados_roc)
        arquivo.write("{}  {} {} {} {} {} \n".format(nome,str(resultados.mean()),str(resultados1.mean()),str(resultados2.mean()),str(resultados3.mean()),str(resultados4.mean())))
    arquivo.close()
    '''
    
    
    #MAKE ORIGINAL DATA RANGE 
    
    '''
    for modelo, nome in zip([RF], [ "RF"]):
        #Curva_Aprendizado(X_PCA, Ydata, modelo, nome)
        #Make_image(X_PCA,Ydata,nome,modelo, pca, Xnormalizer, variaveis,minerio_esteril)
        Report_Classificacao(X_PCA, Ydata, modelo, nome,writer,minerio_esteril )

    '''
    
    
    for modelo, nome in zip([LOGISTIC, LDA, SVM, RF, NEURAL, KNC, DT], [" LOGISTIC", " LDA", " SVM", " RF", " NEURAL", " KNC", "DT"]):
        Curva_Aprendizado(X_PCA, Ydata, modelo, nome)
        #Make_image(X_PCA,Ydata,nome,modelo, pca, Xnormalizer, variaveis,minerio_esteril)
        Report_Classificacao(X_PCA, Ydata, modelo, nome,writer,minerio_esteril )
    
    

if __name__=="__main__":
    
    # INPUTS DO PROGRAMA 
    endereco_dados = 'OFFSET2.xlsx'
    nome_sheet_dados = 'TODOS  20-45 mm'
    nome_sheet_variaveis = 'Variaveis'
    planilha_saida = 'relatorio.xlsx'
    grupos_name = 'grupos.xlsx'
    count_importance = 0.95
    numero_grupos = 3
    ndesvpad = 5
    minerio_esteril = True
    cut_off_ouro = 0.4
    cut_off_enxofre = 0.2
    
    main(planilha_saida, count_importance, grupos_name, numero_grupos, 
         ndesvpad,nome_sheet_dados, nome_sheet_variaveis, 
         cut_off_ouro, cut_off_enxofre, minerio_esteril)



 