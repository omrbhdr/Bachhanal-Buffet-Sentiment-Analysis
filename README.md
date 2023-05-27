# Bachhanal-Buffet-Sentiment-Analysis
#### ÖDEVDE DİĞER PROJELERDEN FARKLI OLARAK;     SPELL İLE YAZIM YANLIŞI OLAN KELİMELERİ DÜZELTTİM, GOOGLETRANS İLE FARKLI DİLDE OLAN YORUMLARI İNGLİZCEYE ÇEVİRDİM. GERİ KALAN BÜŞRA HANIM İLE CEMİLE HANIMIN ÖDEVLERİNDEN VE NLP KONUSUNDAKİ NOTEBOOKLARDAN YARDIM ALDIM.

### Bacchanal Buffet , the Second most popular and highly rated Las Vegas Restaurant
.............................libs................................

              import pandas as pd
              import matplotlib.pyplot as plt
              import seaborn as sns
              from PIL import Image
              import numpy as np
              import nltk
              from nltk.stem.porter import PorterStemmer
              from nltk.corpus import stopwords
              from wordcloud import WordCloud
              from sklearn.feature_extraction.text import CountVectorizer
              from sklearn.model_selection import train_test_split
              from sklearn.naive_bayes import MultinomialNB
              from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
              from sklearn import model_selection
              from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
              from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report
              from sklearn.preprocessing import scale
              from sklearn.preprocessing import StandardScaler
              from sklearn.neighbors import KNeighborsClassifier
              from sklearn.tree import DecisionTreeClassifier
              from sklearn.linear_model import LogisticRegression
              from sklearn.neural_network import MLPClassifier
              from sklearn.ensemble import RandomForestClassifier
              from sklearn.ensemble import GradientBoostingClassifier
              from sklearn.svm import SVC
              from sklearn.linear_model import LogisticRegression
              import googletrans
              from google_trans_new import google_translator
              from googletrans import Translator
              from textblob import TextBlob
              def stemming(tokenized_text)
              
              
              
#### WORD CLOUD ÇALIŞMASI
              
              from PIL import Image  
              from wordcloud import WordCloud, STOPWORDS
.........................................................................


![image](https://github.com/omrbhdr/Bachhanal-Buffet-Sentiment-Analysis/assets/12261537/32aa2481-7b04-4a78-8fcf-de06a47b0b42)


   ## NLP Projesi için yapmamız gerekenler

          1) Tüm cümleleri küçük harfe çevirmek

          2) Noktalma işaretlerini kaldır

          3) Rakamları kaldır

          4) Satır sonunda enter a basılmışsa onu kaldır \n gibi

          5) Gereksiz kelimeleri çıkart - stopwordleri çıkartmak

          6) Tokenize işlemini yap. yani her cümleyi kelimelere ayır

          7) Lemma ve Stemma yani eki kaldı kökü bul

          8) Vectorrizer işlemini yap rakama çeviriyor

![image](https://github.com/omrbhdr/Bachhanal-Buffet-Sentiment-Analysis/assets/12261537/093fd7c5-206d-47b2-a73e-25d03aa1e766)

