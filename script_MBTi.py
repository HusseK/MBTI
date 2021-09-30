from MBTI import MBTI
import pickle

MBTI=MBTI()
    
if MBTI.DataUnit.pdata is None:
    MBTI.DataUnit.preprocess()
else:
    MBTI.DataUnit.data=MBTI.DataUnit.pdata
    
MBTI.DataUnit.lematize_with_pos()

with open('processed_data_lem_pos.pkl', 'wb') as f1:
    pickle.dump(MBTI.DataUnit.data, f1)
    
    
