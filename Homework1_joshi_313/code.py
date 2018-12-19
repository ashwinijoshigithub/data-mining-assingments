#importing required packages
import numpy as np
import csv
import math
import collections
from collections import Counter
from itertools import groupby

#User input K
k = int(input("please enter the value of k:"))

#reading income dataset from CSV
with open('income_tr.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    data = {}
    for row in reader:
        for header, value in row.items():
          try:
            data[header].append(value)
          except KeyError:
            data[header] = [value]
    ID = data['ID']
    age = data['age']
    workclass = data['workclass']
    fnlwgt = data['fnlwgt']
    edu = data['education_cat']
    m_status = data['marital_status']
    occupation = data['occupation']
    rship = data['relationship']
    race = data['race']
    gender = data['gender']
    gain = data['capital_gain']
    loss = data['capital_loss']
    hours = data['hour_per_week']
    country = data['native_country']


    #Normalizing continuous data attributes in range zero to one
    #AGE
    age = [float(numeric_string) for numeric_string in data['age']]
    max_age = max(age)
    min_age = min(age)
    diff = max_age - min_age
    for i in range (0, 520):
        age[i] = (age[i] - min_age) / (diff)

    #FNLWGT
    fnlwgt = [float(numeric_string) for numeric_string in data['fnlwgt']]
    max_wt = max(fnlwgt)
    min_wt = min(fnlwgt)
    diff = max_wt - min_wt
    for i in range (0, 520):
        fnlwgt[i] = (fnlwgt[i] - min_wt) / (diff)

    #Capital Gain    
    gain = [float(numeric_string) for numeric_string in data['capital_gain']]
    max_gain = max(gain)
    min_gain = min(gain)
    diff = max_gain - min_gain
    for i in range (0, 520):
        gain[i] = (gain[i] - min_gain) / (diff)

    #Capital Loss
    loss = [float(numeric_string) for numeric_string in data['capital_loss']]
    max_loss = max(loss)
    min_loss = min(loss)
    diff = max_loss - min_loss
    for i in range (0, 520):
        loss[i] = (loss[i] - min_loss) / (diff)

    #Hours per Week
    hours = [float(numeric_string) for numeric_string in data['hour_per_week']]
    max_hours = max(hours)
    min_hours = min(hours)
    diff = max_hours - min_hours
    for i in range (0, 520):
        hours[i] = (hours[i] - min_hours) / (diff)


    #Filling out missing values with the most frequent value of the attribute
    #Occupation - Filling missing values with 'Adm-Clerical'    
    occuptn = np.array(data['occupation'])
    occupation_list = collections.Counter(occuptn)
    occupation_sorted = occupation_list.most_common()
    b = 0
    for i in occupation:
        for j in occupation_sorted:
            if('?' in i or 'NULL' in i):
                occupation[b] = occupation_sorted[0][0]
        b += 1

    #Workclass - Missing values - Filling with 'Private'
    wkclass = np.array(data['workclass'])
    workclass_list = collections.Counter(wkclass)
    workclass_sorted = workclass_list.most_common()
    b = 0
    for i in workclass:
        if('?' in i or 'NULL' in i):
            workclass[b] = workclass_sorted[0][0]
        b += 1

    #Native_country - Filling values with 'United-states'
    cntry = np.array(data['native_country'])
    country_list = collections.Counter(cntry)
    country_sorted = country_list.most_common()
    b = 0
    for i in country:
        if('?' in i or 'NULL' in i):
            country[b] = country_sorted[0][0]
        b += 1

    #Putting all the attributes into a 2D array where each row represents a record in the dataset
    rec = np.empty(shape=(520, 13))
    arr = [age, workclass, fnlwgt, edu, m_status, occupation, rship, race, gender, gain, loss, hours, country]
    new_arr = zip(*arr)
    rec = np.array(new_arr)
    
    #Creating an array for storing calculated proximities
    dist = np.empty(shape=(520,520), dtype=float)

    #Calculating distinct values in the set of ordinal attributes for proximity calculation
    len_edu = len(set(edu))
    
    #calculating proximity with general approach
    for i in range (0, 520):
        for j in range (0, 520):
            #Distance between a record with itself is of no use so assiging a huge number to it
            if(i == j):
                dist[i][j] = 5000000
            else:
                #Age - continuous
                d_age = abs(float(rec[i][0]) - float(rec[j][0]))
                s_age = 1 / (1 + d_age)
                delta_age = 1
                weight_age = s_age * delta_age

                #workclass - nominal
                if (rec[i][1] == rec[j][1]):
                    s_workclass = 1
                    delta_workclass = 1
                else:
                    s_workclass = 0
                    delta_workclass = 1
                weight_workclass = s_workclass * delta_workclass

                #fnlwgt continuous
                d_fnlwgt = abs(float(rec[i][2]) - float(rec[j][2]))
                s_fnlwgt = 1 / (1 + d_fnlwgt)
                delta_fnlwgt = 1
                weight_fnlwgt = s_fnlwgt * delta_fnlwgt

                #education category - ordinal
                d_edu = abs((float(rec[i][3]) - float(rec[j][3]))) / (len_edu - 1)
                s_edu = 1 - d_edu
                delta_edu = 1
                weight_edu = s_edu * delta_edu

                #marital status - nominal
                if (rec[i][4] == rec[j][4]):
                    s_marital = 1
                else:
                    s_marital = 0
                delta_marital = 1
                weight_marital = s_marital * delta_marital

                #occupation - nominal
                if (rec[i][5] == rec[j][5]):
                    s_occupation = 1
                else:
                    s_occupation = 0
                delta_occupation = 1
                weight_occupation = s_occupation * delta_occupation

                #relationship - nominal
                if (rec[i][6] == rec[j][6]):
                    s_rship = 1
                else:
                    s_rship = 0
                delta_rship = 1
                weight_rship = s_rship * delta_rship

                #race - nominal
                if (rec[i][7] == rec[j][7]):
                    s_race = 1
                else:
                    s_race = 0
                delta_race = 1
                weight_race = s_race * delta_race

                #gender - nominal
                if (rec[i][8] == rec[j][8]):
                    s_gender = 1
                else:
                    s_gender = 0
                delta_gender = 1
                weight_gender = s_gender * delta_gender

                #capital gain - continuous
                d_gain = abs(float(rec[i][9]) - float(rec[j][9]))
                s_gain = 1 / (1 + d_gain)
                delta_gain = 1
                weight_gain = s_gain * delta_gain

                #capital loss - continuous
                d_loss = abs(float(rec[i][10]) - float(rec[j][10]))
                s_loss = 1 / (1 + d_loss)
                delta_loss = 1
                weight_loss = s_loss * delta_loss

                #hours per week - continuous
                d_hours = abs(float(rec[i][11]) - float(rec[j][11]))
                s_hours = 1 / (1 + d_hours)
                delta_hours = 1
                weight_hours = s_hours * delta_hours

                #native country - nominal
                if (rec[i][12] == rec[j][12]):
                    s_country = 1
                    delta_country = 1
                else:
                    s_country = 0
                    delta_country = 1
                weight_country = s_country * delta_country

                #calculating net weight of all the attributes divided by sum of their deltas give similarity and subtracting from 1 gives dissimilarity
                net_weight = weight_age + weight_workclass + weight_fnlwgt + weight_edu + weight_marital + weight_occupation + weight_rship + weight_race + weight_gender + weight_gain + weight_loss + weight_hours + weight_country
                delta_total = delta_age + delta_workclass + delta_fnlwgt + delta_edu + delta_marital + delta_occupation + delta_rship + delta_race + delta_gender + delta_gain + delta_loss + delta_hours +delta_country
                dist[i][j] = 1 - (net_weight / delta_total)
                
    #preparing output for writing into a csv file
    heading_list = []
    for j in range (0, 2*k+1):
        if (j == 0):
            heading_list.insert(j, 'Record Number(ID)')
        elif (j == 1):
            heading_list.insert(j, '1st(ID)')
        elif (j == 2):
            heading_list.insert(j, '1st Prox')
        elif (j == 3):
            heading_list.insert(j, '2nd(ID)')
        elif ( j == 4):
            heading_list.insert(j, '2nd Prox')
        elif (j == 5):
            heading_list.insert(j, '3rd(ID)')
        elif ( j == 6):
            heading_list.insert(j, '3rd Prox')
        elif ( j%2 == 1):
            heading_list.insert(j, str((j/2)+1) + 'th(ID)')
        else:
            heading_list.insert(j, str(j/2) + 'th Prox')

                
    list_final = [[0 for j in range (2*k+1)] for i in range (520)]
    for i in range (0, 520):
        ind = dist[i].argsort()
        id_i = int(ID[i])
        list_final[i][0] = str(i + 1) #+ '(' + str(id_i) + ')'
        b = 1
        for j in ind:
            if (b == 2*k+1):
                break
            else:
                id_j = int(ID[j])
                list_final[i][b] = str(j + 1) #+ '(' + str(id_j) + ')'
                list_final[i][b+1] = dist[i][j]
            b += 2

    #writing to csv ile
    with open('output_general_joshi_313.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(list_final)

    #normalizing attribute 'education' for calculating proximity with Euclidean Distance
    for i in range(0, 520):
        edu[i] = float((float(edu[i]) - 1) / len_edu)

    #updating the records with new values of normalized education values
        
    #one for continuous attributes
    rec_cont = np.empty(shape=(520, 6))
    arr_cont = [age, fnlwgt, edu, gain, loss, hours]
    new_arr_cont = zip(*arr_cont)
    rec_cont = np.array(new_arr_cont)

    #one for nominal attributes
    rec_nom = np.empty(shape=(520, 7))
    arr_nom = [workclass, m_status, occupation, rship, race, gender, country]
    new_arr_nom = zip(*arr_nom)
    rec_nom = np.array(new_arr_nom)

    #Creating an array for storing calculated proximities
    dist_new = np.empty(shape=(520,520), dtype=float)

    #implementing Eculidean Distance between rows of the data set
    def dist_fun(x,y, a, b):   
        cont_sum = 0
        for i in range (0, 6):
            sqr = (x[i]-y[i]) * (x[i]-y[i])
            cont_sum = cont_sum + sqr
        nom_sum = 0
        for i in range (0, 7):
            if (a[i] != b[i]):
                nom_sum += 1
        return math.sqrt(nom_sum + cont_sum)
            
    for i in range (0, 520):
        for j in range (0, 520):
            if (i == j):
                dist_new[i][j] = 10000000
            else:
                dist_new[i][j] = dist_fun(rec_cont[i], rec_cont[j], rec_nom[i], rec_nom[j])

    #preparing output for csv file
    list_final_new = [[0 for j in range (2*k+1)] for i in range (520)]
    for i in range (0, 520):
        ind = dist_new[i].argsort()
        id_i = int(ID[i])
        list_final_new[i][0] = str(i + 1) #+ '(' + str(id_i) + ')'
        b = 1
        for j in ind:
            if (b == 2*k+1):
                break
            else:
                id_j = int(ID[j])
                list_final_new[i][b] = str(j + 1) #+ '(' + str(id_j) + ')'
                list_final_new[i][b+1] = dist_new[i][j]
            b += 2

    #writing to csv file
    with open('output_eucld_joshi_313.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(list_final_new)
            



