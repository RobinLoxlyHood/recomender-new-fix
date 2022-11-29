import os
import pymysql
import mysql.connector
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tensorflow as tf
from tensorflow import keras
from RecommendationModel import RecommenderNet

def get_DataRating():
    my_conn = mysql.connector.connect(host="127.0.0.1",
                                        port=3306,
                                        user='root',
                                        # password='f#Ur8J3N',
                                        database='tour_in')
    query_insert="""
    SELECT * FROM rating_fe;
    """
    raw_rating_fe = pd.read_sql_query(query_insert, my_conn)
    return raw_rating_fe

def get_DataRatingbyID(test):
    my_conn = mysql.connector.connect(host="127.0.0.1",
                                        port=3306,
                                        user='root',
                                        # password='f#Ur8J3N',
                                        database='tour_in')
    query_insert="""
    SELECT * FROM rating_fe;
    """
    raw_rating_fe = pd.read_sql_query(query_insert, my_conn)
    return raw_rating_fe

def insert_to_list_rekomen_db(rp2):
    # Connect to the database
    connection = pymysql.connect(host="127.0.0.1",
                                 port=3306,
                                 user='root',
                                 # password='f#Ur8J3N',
                                 database='tour_in')
    # create cursor
    cursor=connection.cursor()
    
    cols = "`,`".join([str(i) for i in rp2.columns.tolist()])
    for i,row in rp2.iterrows():
        sql = "INSERT INTO `list_rekomen` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))

        # the connection is not autocommitted by default, so we must commit to save our changes
        connection.commit()

def delete_list_rekomen(test):
    # Connect to the database
    connection = pymysql.connect(host="127.0.0.1",
                                 port=3306,
                                 user='root',
                                 # password='f#Ur8J3N',
                                 database='tour_in')
    # create cursor
    cursor=connection.cursor()    
    cursor.execute("DELETE FROM `list_rekomen` WHERE `kode_user` = %s", (test))
    connection.commit()

def get_DataListWisata():
    my_conn = mysql.connector.connect(host="127.0.0.1",
                                        port=3306,
                                        user='root',
                                        # password='f#Ur8J3N',
                                        database='tour_in')
    query_insert="""
    SELECT * FROM list_wisata;
    """
    raw_list_wisata = pd.read_sql_query(query_insert, my_conn)
    return raw_list_wisata




   
def training(df):  
    def dict_encoder(col, data=df):
        # Mengubah kolom suatu dataframe menjadi list tanpa nilai yang sama
        unique_val = data[col].unique().tolist()

        # Melakukan encoding value kolom suatu dataframe ke angka
        val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    
        # Melakukan proses encoding angka ke value dari kolom suatu dataframe
        val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}    
        return val_to_val_encoded, val_encoded_to_val
        #Encoding dan Mapping Kolom User
               
    # Encoding User_Id
    user_to_user_encoded, user_encoded_to_user = dict_encoder('id_user')

    # Mapping User_Id ke dataframe
    df['user'] = df['id_user'].map(user_to_user_encoded)

    # Encoding dan Mapping Kolom Place

    # Encoding Place_Id
    place_to_place_encoded, place_encoded_to_place = dict_encoder('id_wisata')

    # Mapping Place_Id ke dataframe place
    df['place'] = df['id_wisata'].map(place_to_place_encoded)

    #setting parameters for model
    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)
    
    # Mengubah rating menjadi nilai float
    df['rating'] = df['rating'].values.astype(np.float32)
    
    # Mendapatkan nilai minimum dan maksimum rating
    min_rating, max_rating = min(df['rating']), max(df['rating'])
    
    # Membuat variabel x untuk mencocokkan data user dan place menjadi satu value
    x = df[['user', 'place']].values
    
    # Membuat variabel y untuk membuat rating dari hasil 
    y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    
    # Membagi menjadi 80% data train dan 20% data validasi
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    model = RecommenderNet(num_users, num_place) # inisialisasi model
    
    # model compile
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    checkpoint = tf.train.Checkpoint(model)
            
    # Training Model
    history = model.fit(x = x_train,
                        y = y_train,
                        batch_size=16,
                        epochs = 50,
                        verbose=1,
                        validation_data = (x_val, y_val)
                       )
    
    
    # delette old model
    mypath = "./model_checkpoint/" #Enter your path here
    for root, dirs, files in os.walk(mypath, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))

        # Add this block to remove folders
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
            
    #save model new    
    return model.save_weights('model_checkpoint/model_weights', save_format='tf')

def testing(test, df, place_df):
    def dict_encoder(col, data=df):
        # Mengubah kolom suatu dataframe menjadi list tanpa nilai yang sama
        unique_val = data[col].unique().tolist()

        # Melakukan encoding value kolom suatu dataframe ke angka
        val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    
        # Melakukan proses encoding angka ke value dari kolom suatu dataframe
        val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}    
        return val_to_val_encoded, val_encoded_to_val
    #Encoding dan Mapping Kolom User

    # Encoding User_Id
    user_to_user_encoded, user_encoded_to_user = dict_encoder('id_user')

    # Mapping User_Id ke dataframe
    df['user'] = df['id_user'].map(user_to_user_encoded)

    # Encoding dan Mapping Kolom Place

    # Encoding Place_Id
    place_to_place_encoded, place_encoded_to_place = dict_encoder('id_wisata')

    # Mapping Place_Id ke dataframe place
    df['place'] = df['id_wisata'].map(place_to_place_encoded)

    #setting parameters for model
    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)



    #Loading model and saved weights
    loaded_model = RecommenderNet(num_users, num_place)
    loaded_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                         optimizer=keras.optimizers.Adam(learning_rate=0.001))
    loaded_model.load_weights('model_checkpoint/model_weights')


    # Inputan User yang akan dicari recomendasinya
    user_id = test.id_user.iloc[0]
    place_visited_by_user = df[df.id_user == user_id]

    # Membuat data lokasi yang belum dikunjungi user
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.id_wisata.values)]['id'] 
    place_not_visited = list(
        set(place_not_visited)
        .intersection(set(place_to_place_encoded.keys()))
    )
    if place_not_visited != []:
        place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
        user_encoder = user_to_user_encoded.get(user_id)
        user_place_array = np.hstack(
            ([[user_encoder]] * len(place_not_visited), place_not_visited)
        )

        # Mengambil top 7 recommendation
        ratings = loaded_model.predict(user_place_array).flatten()
        top_ratings_indices = ratings.argsort()[-9:][::-1]
        recommended_place_ids = [
            place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
        ]

        #get the anime_id's, so we can make a request to anilist api for more info on show
        recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    
        return recommended_place
    else:
        kode_wisata={"id":[101,102,103,104,105,106,107,108,109,110]}
        df1 = pd.DataFrame(data = kode_wisata)
        
        return df1
        
 
    
    