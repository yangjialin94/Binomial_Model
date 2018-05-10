
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pandas_datareader.data import Options


#############################################################################################  
##################################### Build Option Trees ####################################
############################################################################################# 

# European Option Tree
# S0: initial stock price 
# u: Probability of Price Increasing
# d: Probability of Price Decreasing
# r: Interest Rate
# K: Strike Price
# N: Terminal Time
# op_name: 'American' or 'European' 
# op_type: 'put' or 'call'
def option_tree(S0, K, N, op_name, op_type):
    
    # Get data from google and store into list stock_list
    start = datetime(2016,11,10)
    end = datetime(2017,11,11)
    d = web.DataReader('GOOG', 'yahoo', start, end)
    stock_list = []      # List with all the Stock Price
    
    for x in d['Close']:
        stock_list.append(x)

################################ Stock Price With Calibration ################################ 
#     # Calibration on Stock Prices
#     mean = sum(stock_list) / len(stock_list)
#     sigma = 0
#      
#     for x in stock_list:
#         sigma += (x-mean)**2
#          
#     sigma = np.sqrt(sigma / (len(stock_list) - 1))
#     cal = []    # List with all the Stock Price after Calibration
#      
#     for x in stock_list:
#         if x >= (mean - 2*sigma) and x <= (mean + 2*sigma):
#             cal.append(x)
#         else:
#             cal.append(mean)
#      
#     stock_list = cal
##############################################################################################
 
    # Calculate u and d
    ret = []      # Returns
     
    for i in range(len(stock_list) - 1):
        ret.append(np.log(stock_list[i+1] / stock_list[i]))
     
    rAvg = sum(ret) / len(ret)      # Average Return
    s = 0
     
    for x in ret:
        s += (x-rAvg)**2
     
    sigmaDaily = np.sqrt(s / (len(ret) - 1))
    sigma = sigmaDaily * np.sqrt(252)
     
    t = 212     # Days between 11/11/2017 and 06/11/2018
    T = t/365.
    Delta_t = T/N
    u = np.exp(sigma * np.sqrt(Delta_t))
    d = np.exp(-sigma * np.sqrt(Delta_t))
     
    # Calculate p and q from r
#     rYear = 0.000187    # 1 Tear LIBOR rate this week
#     r = 0.0187 /365 * Delta_t
    r = 0
#     p = (1+r-d) / (u-d)
#     q = 1 - p  
    
    # Create parameters
    stock_tree = np.zeros((N+1, N+1))    # Matrix with stock prices, column number represent the time period
    option_tree = np.zeros((N+1, N+1))   # Matrix with option (call/put) price at each node, column number represent the time period
    hedge_tree = np.zeros((N, N), dtype='i,i').tolist()  # Matrix with hedging strategy at each node, column number represent the time period
    
    # Fill out stock_tree
    for i in range(N+1):    # Loop over row
        for j in range(i, N+1):     # Loop over column
            stock_tree[i][j] = S0 * float(u)**(j-i) * float(d)**(i)
           
    # Fill out the profit at T
    for i in range(N+1):
        if op_type.lower() == 'call':    # Profit for call option
            option_tree[i][N] = max(stock_tree[i][N] - K, 0)
        elif op_type.lower() == 'put':   # Profit for put option
            option_tree[i][N] = max(K - stock_tree[i][N], 0)
        else:
            print('Invalid Option Type')    # Error message for invalid option types
            return
    
    # Fill out option_tree and hedge_tree
    for j in range(N, 0, -1):  # Loop over column backward
        for i in range(j):    # Loop over row
            H1 = float(option_tree[i][j] - option_tree[i+1][j]) / float(stock_tree[i][j] - stock_tree[i+1][j])
            H0 = float(option_tree[i][j] - H1 * stock_tree[i][j]) / float(1+r)
            hedge_tree[i][j-1] = (H0 ,H1)
            
            if op_name.lower() == 'european':    # Price for European option
                option_tree[i][j-1] = H0 + H1 * stock_tree[i][j-1]
            elif op_name.lower() == 'american':    # Price for American option
                if op_type.lower() == 'call':    # For call option
                    option_tree[i][j-1] = max(H0 + H1 * stock_tree[i][j-1], stock_tree[i][j-1] - K)
                elif op_type.lower() == 'put':   # For put option
                    option_tree[i][j-1] = max(H0 + H1 * stock_tree[i][j-1], K - stock_tree[i][j-1])
            else:
                print('Invalid Option Name')    # Error message for invalid option name
                return
    
    # return stock_tree, option_tree, and hedge_tree
    return stock_tree, option_tree, hedge_tree

# # Test
# S0 = float(raw_input('Please enter initial Stock Price (S0): '))
# K = float(raw_input('Please enter Strike Price (K): '))
# N = int(raw_input('Please enter Number of Period (N): '))
# op_name = raw_input('Please enter Option Name (European/American): ')
# op_type = raw_input('Please enter Option Type (call/put): ')
# stock, option, hedge = option_tree(S0, K, N, op_name, op_type)

# stock, option, hedge = option_tree(10, 10, 2, 'ameRican', 'Put')
# print(stock)
# print(option)
# print(hedge)


#############################################################################################  
##################################### Output the Result #####################################
############################################################################################# 

# Get Bid Price, Ask Price for Call and Put Options
goog = Options('GOOG', 'yahoo')
call_data = goog.get_call_data(6, 2018)
bid_call = call_data['Bid']
ask_call = call_data['Ask']
put_data = goog.get_put_data(6, 2018)
bid_put = put_data['Bid']
ask_put = put_data['Ask']

# Calculate the Value of Option
# value_call_option = []    # Value of Option for Call Option
# value_put_option = []     # Value of Option for Put Option
# 
# for i in range(len(bid_call)):
#     value_call_option.append((bid_call[i] + ask_call[i]) / 2)
#     value_put_option.append((bid_put[i] + ask_put[i]) / 2)

# Get Options' Strike Prices for Call and Put
call_data.to_csv('fileCall.csv')
temp_call_file = pd.read_csv('fileCall.csv')
strike_call = temp_call_file['Strike']    # List of Strike Prices for Call Options
put_data.to_csv('filePut.csv')
temp_put_file = pd.read_csv('filePut.csv')
strike_put = temp_put_file['Strike']    # List of Strike Prices for Put Options

# Used today's Stock Price and Strike Prices to calculate theoretical Values for Options
stock_stock_price = 1041.41     # Today's Google Stock Price
value_call = []      # Value for Call Options
value_put = []       # Value for Put Options
i = 1       # Increment
j = 1       # Increment

############################################# TEST ###########################################
# # Test output for call option's hedge_tree with K=420 and N=3
# stock_call, option_call, hedge_call = option_tree(stock_stock_price, 420, 3, 'european', 'call')
# print(hedge_call)
# 
# # Test output for put option's hedge_tree with K=1000 and N=3
# stock_put, option_put, hedge_put = option_tree(stock_stock_price, 1000, 3, 'european', 'put')
# print(hedge_put)
##############################################################################################
  
for x in strike_call:
    stock_call, option_call, hedge_call = option_tree(stock_stock_price, x, 200, 'european', 'call')
    value_call.append(option_call[0][0])
    print('{}: Strike Price {} for Call is Done.'.format(i,x))
    i = i+1
  
for x in strike_put:
    stock_put, option_put, hedge_put = option_tree(stock_stock_price, x, 200, 'european', 'put')
    value_put.append(option_put[0][0])
    print('{}: Strike Price {} for Put is Done.'.format(j,x))
    j = j+1
 
############################### Output Value Without Calibration ##############################
with open('theoCall.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(value_call)
    print('Output theoCall.csv DONE.')
      
with open('theoPut.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(value_put)
    print('Output theoPut.csv DONE.')

################################ Output Value With Calibration ###############################
# with open('theoCalibCall.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(value_call)
#     print('Output theoCalibCall.csv DONE.')
#     
# with open('theoCalibPut.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(put_price)
#     print('Output theoCalibPut.csv DONE.')


#############################################################################################  
###################################### Plot the Graphs ######################################
#############################################################################################  

######################################## CALL OPTION ########################################
# Parameters
file_call_graph = pd.read_csv('fileCall.csv')
strike_call_graph = file_call_graph['Strike']    # List of Strike Prices for Call Options
bid_call_graph = file_call_graph['Bid']    # List of Bid Prices for Call Options
ask_call_graph = file_call_graph['Ask']    # List of Ask Prices for Call Options
value_call_graph = (bid_call_graph + ask_call_graph) / 2.     # List of Values for Call Options
 
# Get Theoretical Call Values
with open("theoCall.csv", 'r') as file:
    read = csv.reader(file)
    value_theo_call = list(read)
 
value_theo_call = np.array(value_theo_call)
value_theo_call = np.transpose(value_theo_call)
 
# Get Calibrated Theoretical Call Values
with open("theoCalibCall.csv", 'r') as file:
    read = csv.reader(file)
    value_calib_theo_call = list(read)
 
value_calib_theo_call = np.array(value_calib_theo_call)
value_calib_theo_call = np.transpose(value_calib_theo_call)
 
# Plot the Graph for Call Options
plt.figure('CALL OPTION')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('Graph for Value of Call Option')
plt.plot(strike_call_graph, value_call_graph, 'blue', label = 'Real Call Value')
plt.plot(strike_call_graph, value_theo_call, 'red', label = 'Theoretical Call Value')
plt.plot(strike_call_graph, value_calib_theo_call, 'green', label = 'Calibrated Theoretical Call Value')
plt.legend(loc=1)
plt.show()
 
######################################## PUT OPTION ########################################
# Parameters
file_put_graph = pd.read_csv('filePut.csv')
strike_price_graph = file_put_graph['Strike']    # List of Strike Prices for Put Options
bid_put_graph = file_put_graph['Bid']    # List of Bid Prices for Put Options
ask_put_graph = file_put_graph['Ask']    # List of Ask Prices for Put Options
value_put_graph = (bid_put_graph + ask_put_graph) / 2.     # List of Values for Put Options
 
# Get Theoretical Put Values
with open("theoPut.csv", 'r') as file:
    read = csv.reader(file)
    value_theo_put = list(read)
 
value_theo_put = np.array(value_theo_put)
value_theo_put = np.transpose(value_theo_put)
 
# Get Calibrated Theoretical Put Values
with open("theoCalibPut.csv", 'r') as file:
    read = csv.reader(file)
    value_calib_theo_put = list(read)
 
value_calib_theo_put = np.array(value_calib_theo_put)
value_calib_theo_put = np.transpose(value_calib_theo_put)
 
# Plot the Graph for Put Options
plt.figure('PUT OPTION')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('Graph for Value of Put Option')
plt.plot(strike_price_graph, value_put_graph, 'blue', label = 'Real Put Value')
plt.plot(strike_price_graph, value_theo_put, 'red', label = 'Theoretical Put Value')
plt.plot(strike_price_graph, value_calib_theo_put, 'green', label = 'Calibrated Theoretical Put Value')
plt.legend(loc=2)
plt.show()




