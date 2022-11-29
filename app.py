import pandas as pd
import numpy as np
from function import *
from datetime import datetime
from RecommendationModel import RecommenderNet
import warnings
warnings.filterwarnings('ignore')


# app = Flask(__name__)
# # @app.route("/")
# # def home():
# #     return "Hello, Flask!"
    

# @app.route("/")
# def recomendation():
#     time_to_run=datetime.today()
#     time_to_run=time_to_run.strftime('%H:%M')
#     if time_to_run == "14:22":
#         ranting_fe=get_DataRating().copy()
#         rating=ranting_fe[['kode_user','kode_wisata','rate_value']]
#         rating.columns = [['id_user','id_wisata', 'rating']]
#         rating.to_csv('Dataset/rating_fe.csv', index=False)
        
#         list_wisata=get_DataListWisata().copy()
#         place=list_wisata[['id_wisata','wisata']]
#         place.columns = ['id','place_name']
#         place.to_csv('Dataset/list_wisata_db.csv', index=False)

#         rating = pd.read_csv('Dataset/rating_fe.csv')
#         place = pd.read_csv('Dataset/list_wisata_db.csv')
#         df=rating.copy()
#         training(df)
#         #testing
    
#         ranting_fe.set_index("time", inplace=True)
#         #getdatetoday
#         today = date.today()
#         today=today.strftime('%Y-%m-%d')
#         test=ranting_fe.loc[today]
#         test=test.iloc[:, 1:2]
#         test=test.reset_index(drop=True)
#         test=test.kode_user.unique()
#         for i in test:
#             test=pd.DataFrame([i], columns=['id_user'])
#             df=rating.copy()
#             place_df=place.copy()
#             result=testing(test, df, place_df)
#             rp2= result.copy()
#             test = test.id_user.iloc[0]
#             kode_user=[]
#             for i in range(len(rp2)):
#                 kode_user.append(test)
#             rp2['kode_user'] = kode_user
#             rp2['id_rek'] = ""
#             rp2 = rp2[["id_rek", "kode_user", "id"]]
#             rp2.columns = ["id_rek", "kode_user", "kode_wisata"]
#             insert_to_list_rekomen_db(rp2)
            
    # return "Selesai"
    
ranting_fe=get_DataRating().copy()
rating=ranting_fe[['kode_user','kode_wisata','rate_value']]
rating.columns = [['id_user','id_wisata', 'rating']]
rating.to_csv('Dataset/rating_fe.csv', index=False)
        
list_wisata=get_DataListWisata().copy()
place=list_wisata[['id_wisata','wisata']]
place.columns = ['id','place_name']
place.to_csv('Dataset/list_wisata_db.csv', index=False)

rating = pd.read_csv('Dataset/rating_fe.csv')
place = pd.read_csv('Dataset/list_wisata_db.csv')
df=rating.copy()
training(df)
#testing
    
ranting_fe.set_index("time", inplace=True)
#getdatetoday
today = date.today()
today=today.strftime('%Y-%m-%d')
test=ranting_fe.loc[today]
test=test.iloc[:, 1:2]
test=test.reset_index(drop=True)
test=test.kode_user.unique()
for i in test:
    test=pd.DataFrame([i], columns=['id_user'])
    df=rating.copy()
    place_df=place.copy()
    result=testing(test, df, place_df)
    rp2= result.copy()
    x=int(i)
    delete_list_rekomen(i)
    test = test.id_user.iloc[0]
    kode_user=[]
    for i in range(len(rp2)):
        kode_user.append(test)
        
    rp2['kode_user'] = kode_user
    rp2['id_rek'] = ""
    rp2 = rp2[["id_rek", "kode_user", "id"]]
    rp2.columns = ["id_rek", "kode_user", "kode_wisata"]       
    insert_to_list_rekomen_db(rp2)    

