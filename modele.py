import pandas as pd 
from sklearn.model_selection import train_test_split #c'est l'outil qui coupe tes données on deux 
from sklearn.ensemble import RandomForestClassifier #c'est le modele lui meme : le foret d'arbres de decision  qui apprend a distinguer un message normal d'une attaque
from sklearn.metrics import classification_report #c'est l'outil qui evalue ton modele . apres les prediction , il affiche les scores pour chaque classe
import joblib
colonnes = ["timest", "can_id","dlc","data0","data1","data2","data3","data4","data5","data6","data7","flag"]
DoS = pd.read_csv("DoS_dataset.csv", header=None , names=colonnes)
Fuzzy=pd.read_csv("Fuzzy_dataset.csv", header=None , names=colonnes)
gear = pd.read_csv("gear_dataset.csv", header=None , names=colonnes)
rpm = pd.read_csv("RPM_dataset.csv", header=None , names=colonnes)
Data = pd.concat([DoS,Fuzzy,gear,rpm],ignore_index=True)
#Data = Data.sample(n=20000000,random_state=42)

Data = Data.sort_values(by="timest").reset_index(drop=True)
Data["delta_temps"] = Data.groupby("can_id")["timest"].diff()
Data["delta_temps"] = Data["delta_temps"].fillna(0)
Data = Data[Data["flag"].isin(["R","T"])]
Data[["data0","data1","data2","data3","data4","data5","data6","data7"]]= Data[["data0","data1","data2","data3","data4","data5","data6","data7"]].fillna("00")
Data["label"]=(Data["flag"]=="T").astype(int) #ajouter une nouvelle colonne qui s'appelle label si est un attaque =1 sinon 0
Data["can_id_num"]=(Data["can_id"]).apply(lambda x : int(str(x),16))
Data["data0_dec"]=(Data["data0"]).apply(lambda x : int(str(x),16))
Data["data1_dec"]=(Data["data1"]).apply(lambda x : int(str(x),16))
Data["data2_dec"]=(Data["data2"]).apply(lambda x : int(str(x),16))
Data["data3_dec"]=(Data["data3"]).apply(lambda x : int(str(x),16))
Data["data4_dec"]=(Data["data4"]).apply(lambda x : int(str(x),16))
Data["data5_dec"]=(Data["data5"]).apply(lambda x : int(str(x),16))
Data["data6_dec"]=(Data["data6"]).apply(lambda x : int(str(x),16))
Data["data7_dec"]=(Data["data7"]).apply(lambda x : int(str(x),16))
X=Data[["can_id_num","data0_dec","data1_dec","data2_dec","data3_dec","data4_dec","data5_dec","data6_dec","data7_dec","delta_temps"]]
Y= Data["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
modele=RandomForestClassifier(n_estimators=50,random_state=42,    class_weight="balanced")#50 est le nombre total des arbre de foret le radom_state fige le hasard chaque arbre prend une partie au hasard du données(data ) et prend la decision selon cette hasard 
modele.fit(X_train,Y_train)
predic = modele.predict(X_test)
print(classification_report(Y_test,predic))
joblib.dump(modele, "modele_can.joblib")
