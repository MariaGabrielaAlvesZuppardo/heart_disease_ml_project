import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self,dataset_id=45):
        self.dataset_id = dataset_id
        self.data = None 
        self.features = None 
        self.target = None
    
    def load_data(self):
        
        heart_disease = fetch_ucirepo(id=self.dataset_id)
        
        #Acessando as features e o target 
        
        self.features = heart_disease.data.features 
        self.target = heart_disease.data.targets 
        
        self.data = pd.DataFrame(self.features, columns = heart_disease.feature_names)
        self.data['target'] = self.target
        
        return self.data
    
    def preprocess_data(self):
        x = self.features
        y = self.target
        
        #Normalizando os dados 
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y 
    
    def split_data(self,X,y, test_size=0.2,random_state=42):
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test