# %%
import set_cwd
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import os
from src.back_end.dataload.database_connector import MicrosoftSQLDBConnector
from src.back_end.dataload.str_query_parser import parse_query
MLDBconn = MicrosoftSQLDBConnector('MLDB').connect()


# %%
# close connection
MLDBconn.close()

# %%
from datetime import datetime, timedelta
columns = ["msg_id", "msg", "chat_id", "msg_datetime", "productID", "dep_id",  "country_code", "wait_time", "chat_duration", "subject", "totalMsgsOp", "totalMsgsUser", "totalMsgsUser", "waitFirstClick"]
data = pd.DataFrame()
check_data = {} #2 if 300 row of data is avialbele, 1 if less than 300 and 0 if none
final_date = '2023-01-01'
start_date = '2022-04-27'
d1 = datetime.strptime(start_date, "%Y-%m-%d")
d2 = datetime.strptime(final_date, "%Y-%m-%d")
# difference between dates in timedelta
number_of_days = d2 - d1
print(f'Difference is {number_of_days.days} days')
i = 0
while i <= number_of_days.days:
    query = parse_query(f"""
DECLARE @start_date date = '{start_date}';
DECLARE @sample_date date = DATEADD(day, {i}, @start_date);
select m.msg_id, m.msg, m.user_id, m.msg_datetime, c.chat_id, c.productID, c.country_code, c.wait_time, c.chat_duration, c.subject, c.totalMsgsOp, c.totalMsgsUser, c.waitFirstClick, c.customerID, c.createdDate --count(distinct(m.chat_id))
from  LHchatMsg m join (select top 100 chat_id, productID, country_code, wait_time, chat_duration, subject, totalMsgsOp, totalMsgsUser, waitFirstClick, customerID, createdDate from LHchats
where  month(createdDate) = month(@sample_date) and day(createdDate) = day(@sample_date) and year(createdDate) = year(@sample_date)) as c
on c.chat_id = m.chat_id
order by c.chat_id
        """)
    day_data = pd.read_sql_query(query, MLDBconn)
    d1 += timedelta(days = i)
    check_data[datetime.strftime(d1, "%Y-%m-%d")] = len(day_data)
    i+=1
    data = pd.concat([day_data, data])
data.head()

# %%
data[data.isna().customerID == False]

# %%
# save this data
data.to_csv('data/prod_v1/chat_data_v1.csv', index= False)

# %%
# save this data
data.to_csv('data/smart_compose/smart_compose_row.csv', index= False)

# %%


# %%
