import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Gender:str,
                 Age:float,
                 Height:float,
                 Weight:float,
                 family_history_with_overweight:str,
                 FAVC:str,
                 FCVC:float,
                 NCP:float,
                 CAEC:str,
                 SMOKE:str,
                 CH2O:float,
                 SCC:str,
                 FAF:float,
                 TUE:float,
                 CALC:str,
                 MTRANS:str
                 
                 ):
        
        self.Gender=Gender
        self.Age=Age
        self.Height=Height
        self.Weight=Weight
        self.family_history_with_overweight=family_history_with_overweight
        self.FAVC=FAVC
        self.FCVC = FCVC
        self.NCP = NCP
        self.CAEC = CAEC
        self.SMOKE=SMOKE
        self.CH2O = CH2O
        self.SCC = SCC
        self.FAF = FAF
        self.TUE = TUE
        self.CALC = CALC
        self.MTRANS = MTRANS

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender':[self.Gender],
                'Age':[self.Age],
                'Height':[self.Height],
                'Weight':[self.Weight],
                'family_history_with_overweight':[self.family_history_with_overweight],
                'FAVC':[self.FAVC],
                'FCVC':[self.FCVC],
                'NCP':[self.NCP],
                'CAEC':[self.CAEC],
                'SMOKE':[self.SMOKE],
                'CH2O':[self.CH2O],
                'SCC':[self.SCC],
                'FAF':[self.FAF],
                'TUE':[self.TUE],
                'CALC':[self.CALC],
                'MTRANS':[self.MTRANS],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
