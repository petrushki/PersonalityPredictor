import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier



def load_data():
    df = pd.read_csv("data\\Personality_Syncora_Synthetic.csv")

    df.columns = df.columns.str.lower()

    numerical_features = ['time_spent_alone', 'social_event_attendance','going_outside','friends_circle_size','post_frequency']

    df = df.replace(-1,np.nan)

    for f in numerical_features:    
        df[f] = df[f].fillna(df[f].median())
        
    return df


def train():
    df_train = load_data()
    y_train = df_train.personality.values
    del df_train['personality']
    X_train = df_train.values
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_leaf': 1
    }
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    
    return model
    

def save_model(model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(model, f_out)
        

model = train()
save_model(model, 'model.bin')

print('Model trained and saved to model.bin')