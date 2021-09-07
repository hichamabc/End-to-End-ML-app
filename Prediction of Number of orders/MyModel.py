from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd


class My_Model:
    variable=0
    def __init__(self,base_scaler,n_estimators,max_depth,data):
        self.scaler=base_scaler
        self.model=GradientBoostingRegressor(n_estimators=n_estimators,max_depth=max_depth,random_state=1)
        self.data=data
        
    def fit(self):
        data=self.data
        X_train=data.loc[:,['week','center_id','meal_id','emailer_for_promotion','homepage_featured','checkout_price','base_price']]
        A=X_train.loc[:,['checkout_price','base_price']]
        scaler=self.scaler
        A_scaled=scaler.fit_transform(A)
        A_df=pd.DataFrame(A_scaled,columns=['checkout_price_scaled','base_price_scaled'])
        data=pd.concat([X_train.loc[:,['week','center_id','meal_id','emailer_for_promotion','homepage_featured']],A_df],axis=1)
        self.scaler=scaler
        return data
    
    def transform(self,data):
        A=data.loc[:,['checkout_price','base_price']]
        scaler=self.scaler
        A_scaled=scaler.transform(A)
        A_df=pd.DataFrame(A_scaled,columns=['checkout_price_scaled','base_price_scaled'])
        data.reset_index(inplace=True)
        data=pd.concat([data.loc[:,['week','center_id','meal_id','emailer_for_promotion','homepage_featured']],A_df],axis=1)
        return data
    
    
    def train(self):
        X_train=self.fit()
        Y_train=self.data['num_orders']
        model=self.model
        model.fit(X_train,Y_train)
        self.model=model
        self.variable=1
    
    
    def predict(self,data):
        if self.variable==1:
            X_test=data.loc[:,['week','center_id','meal_id','emailer_for_promotion','homepage_featured','checkout_price','base_price']]
            X_test=self.transform(X_test)
            model=self.model
            Y_predicted=model.predict(X_test)
            return Y_predicted
        else:
            return "Train the model first !"
    def  mean_squared_error(self,a,b):
        return mean_squared_error(a,b,squared=False)