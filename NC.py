# Load scikit's random forest classifier library
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.datasets import load_iris

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)


classes = ['bangla','oriya','persian','roman']
# df = pd.read_csv("/home/lcorra/Documentos/RP/Base/lpq-octave/vetoresAleatoriolpq.csv")	
# df = pd.read_csv("/home/lcorra/Documentos/RP/Base/lpq-octave/vetoresFixolpq.csv")	
# df = pd.read_csv("/home/lcorra/Documentos/RP/Base/lbp-octave/vetoresAleatoriolbp.csv")
df = pd.read_csv("/home/lcorra/Documentos/RP/Base/lbp-octave/vetoresFixolbp.csv")
rotulos = pd.read_csv("/home/lcorra/Documentos/RP/Base/vetoresAleatorioROTULOS.csv")
c_treino = 0
c_teste = 0
treino=pd.DataFrame(columns = df.columns)
rotulos_treino=pd.DataFrame()
teste=pd.DataFrame()
rotulos_teste=pd.DataFrame()
df['Classes'] = rotulos
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
# print(df[9:18]['C1'])
df['isTrain']=None
i = 0
while(i<len(df)):
	if(np.random.uniform(0,1)<=0.75):
 		df.loc[i:i+9,'isTrain']=True	
	else:
 		df.loc[i:i+9,'isTrain']=False	

				
	i+=9


#print(treino.head())
train,test = df[df['isTrain']==True], df[df['isTrain']==False]
features=df.columns[:len(df.columns)-2]
# print(features[10])
y=pd.factorize(train['Classes'])[0]
# print(train[features])

clf=NearestCentroid()
# print(train[features])
clf.fit(train[features],y)
resultado=clf.predict(test[features])
i=0
tabelinha=[[0 for x in range(len(classes))] for y in range(len(classes))]
# final_fotos=[]
while(i<len(resultado)):
	tabelinha[np.argmax(np.bincount(resultado[i:i+9]))][np.argmax(np.bincount(pd.factorize(test['Classes'])[0][i:i+9]))]= tabelinha[np.argmax(np.bincount(resultado[i:i+9]))][np.argmax(np.bincount(pd.factorize(test['Classes'])[0][i:i+9]))]+1

	i+=9

# i=0

# final_fotos_certo=[]
# while(i<len(pd.factorize(test['Classes'])[0])):
# 	final_fotos_certo.append(np.argmax(np.bincount(pd.factorize(test['Classes'])[0][i:i+9])))

# 	i+=9

# print(final_fotos_certo)
# print(final_fotos)

# preds = classes[clf.predict(test[features])]
# print(clf.predict_proba(test[features]))
# tabelinha=pd.crosstab(pd.factorize(test['Classes'])[0],clf.predict(test[features]),rownames=['Actual Species'], colnames=['Predicted Species'])
#print(len(tabelinha))
i=0
j=0
for i in range(len(tabelinha)):
	print(tabelinha[i])
