import os
from deta import Deta
from dotenv import load_dotenv
import datetime
import pandas as pd
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

deta= Deta(DETA_KEY)
db = deta.Base("clients")
def insert_data(age,job,marital,education, default,housing,loan,contact,month,day_of_week, duration,campaign,pdays, previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed,current_time):
    current_time = datetime.datetime.now().isoformat()
    return db.put({"age":age,"job":job,"marital":marital,"education":education,
                 "default":default,"housing":housing,"loan":loan,
                 "contact":contact,"month":month,"day_of_week":day_of_week,"duration": duration,
                 "campaign":campaign,"pdays":pdays,"previous": previous,
                 "poutcome":poutcome,"emp.var.rate":emp_var_rate,"cons.price.idx":cons_price_idx,
                 "cons.conf.idx":cons_conf_idx,
                 "euribor3m":euribor3m,"nr.employed":nr_employed,"time": current_time})
    
def fetch_all_data():
    res= db.fetch()
    return res.items
