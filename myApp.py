import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

class App:
    def __init__(self):
        self.selected_dataset = None
        self.classifier_name = 'KNN'
        self.data = None
        self.Init_Streamlit_Page()
        self.model = None
        
    def run(self):
        self.get_dataset_and_classifier()
        self.data_preprocessing()
        self.generate()
    def Init_Streamlit_Page(self):
        st.title('Breast Cancer Classification')

        st.write("""
        # Explore different classifier on Breast Cancer dataset
        """)

        self.selected_dataset = st.sidebar.selectbox(
            'Select Dataset',
            ['Empty Dataset','Breast Cancer','Iris', ' Wine']
        )

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naïve Bayes')
        )
        st.write(f" ## {self.classifier_name} classifier")
        
    def get_dataset_and_classifier(self):
        
        if self.selected_dataset != 'Breast Cancer':
            st.error("Lütfen geçerli bir veri seti seçin !")
            exit()
        else:
            self.data = pd.read_csv(r"C:\Users\emre\Projects\data.csv")
            st.subheader("Verinin İlk 10 Satırı")
            st.write(self.data.head(10))

        if self.classifier_name == 'SVM':
            self.model  = SVC()
        elif self.classifier_name == 'KNN':
           self.model  = KNeighborsClassifier()  # default olarak 2 verdim
        else:
            self.model  = GaussianNB()
            
        
    def data_preprocessing(self):
            self.data.drop(['id','Unnamed: 32'], axis = 1, inplace= True )
            st.subheader("Temizlenmiş Verinin Son 10 Satırı")
            st.write(self.data.tail(10))
            
            self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})
            self.y = self.data['diagnosis'] 
            self.X = self.data.drop(['diagnosis'], axis=1)
            
           
            
            correlation_matrix = self.data.corr()
    
            # Korelasyon matrisini çizdirme
            plt.figure(figsize=(25, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Korelasyon Matrisi')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            
            malignant_data = self.data[self.data['diagnosis'] == 1]
            benign_data = self.data[self.data['diagnosis'] == 0]
            
            # x: radiusmean, y: texturemean olarak scatter plot çizdirme
            plt.figure(figsize=(10, 8))
            plt.scatter(malignant_data['radius_mean'], malignant_data['texture_mean'], color='red', label='Malignant')
            plt.scatter(benign_data['radius_mean'], benign_data['texture_mean'], color='blue', label='Benign')
            plt.xlabel('radius_mean')
            plt.ylabel('texture_mean')
            plt.legend()
            st.pyplot()
       
        
    def findOptimumModel(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        
        
        if self.classifier_name == 'SVM':
            param_grid = {
            'C': [0.1, 1, 10]
        }
            
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
            
            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_
            
        elif self.classifier_name == 'KNN':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            self.X_test = sc.transform(self.X_test)
            
            param_grid = {
                'n_neighbors': [1,2,3,4,5,6,7,8,9]
            }
            
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)

            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_
        
        else:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            self.X_test = sc.transform(self.X_test)
            param_grid_nb = {
                    'var_smoothing': np.logspace(0,-9, num=10)
                }       
            grid_search = GridSearchCV(self.model, param_grid_nb, cv=5, scoring='r2', verbose=1, n_jobs=-1)

            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_

    def generate(self):
        
        my_best_model = self.findOptimumModel()
        y_pred = my_best_model.predict(self.X_test)
        
        hata_matrisi = confusion_matrix(self.y_test, y_pred)
        index = ['benign','malignant']
        colums = ['benign','malignant']
        hata_goster = pd.DataFrame(hata_matrisi,index,colums)
        plt.figure(figsize=(15,6))
        sns.heatmap(hata_goster,annot=True)   
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('y_test', fontsize=20,labelpad=50)
        plt.xlabel('y_pred', fontsize=20,labelpad=50)
        st.pyplot()
        
        st.write(f'Accuracy Score =', accuracy_score(self.y_test, y_pred))
        st.write("Precision Score:", metrics.precision_score(self.y_test,y_pred,average='weighted'))
        st.write("Recall Score:", metrics.recall_score(self.y_test,y_pred,average='weighted'))
        st.write("F1 Score:", metrics.f1_score(self.y_test,y_pred,average='weighted'))
