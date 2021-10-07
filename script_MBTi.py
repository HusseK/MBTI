from MBTI import MBTI
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

MBTI=MBTI()
    
if MBTI.DataUnit.pdata is None:
    MBTI.DataUnit.prepare_data()
    MBTI.DataUnit.preprocess()
else:
    MBTI.DataUnit.data=MBTI.DataUnit.pdata
    
MBTI.DataUnit.tokenize_and_create_attention_masks()
    
#MBTI.DataUnit.lematize_with_pos()

#with open('processed_data_lem_pos.pkl', 'wb') as f1:
#    pickle.dump(MBTI.DataUnit.data, f1)




#kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    
