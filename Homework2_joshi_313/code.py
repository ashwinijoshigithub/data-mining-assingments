#importing required packages
import csv
import math
import collections
import numpy as np
from itertools import groupby
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

#setting value of k
k = 8

#reading training dataset from CSV
with open('income_tr.csv') as csvtrfile:
    reader = csv.DictReader(csvtrfile)
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
    class_attr = data['class']

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



#reading test dataset from CSV
with open('income_te.csv') as csvtefile:
    reader = csv.DictReader(csvtefile)
    data_te = {}
    for row in reader:
        for header, value in row.items():
          try:
            data_te[header].append(value)
          except KeyError:
            data_te[header] = [value]
    ID_te = data_te['ID']
    age_te = data_te['age']
    workclass_te = data_te['workclass']
    fnlwgt_te = data_te['fnlwgt']
    edu_te = data_te['education_cat']
    m_status_te = data_te['marital_status']
    occupation_te = data_te['occupation']
    rship_te = data_te['relationship']
    race_te = data_te['race']
    gender_te = data_te['gender']
    gain_te = data_te['capital_gain']
    loss_te = data_te['capital_loss']
    hours_te = data_te['hour_per_week']
    country_te = data_te['native_country']
    class_attr_te = data_te['class']

    #Filling out missing values with the most frequent value of the attribute
    #Occupation - Filling missing values with 'Adm-Clerical'
    occuptn_te = np.array(data_te['occupation'])
    occupation_list_te = collections.Counter(occuptn_te)
    occupation_sorted_te = occupation_list_te.most_common()
    b = 0
    for i in occupation_te:
        if('?' in i or 'NULL' in i):
                occupation_te[b] = occupation_sorted_te[0][0]
        b += 1

    #Workclass - Missing values - Filling with 'Private'
    wkclass_te = np.array(data_te['workclass'])
    workclass_list_te = collections.Counter(wkclass_te)
    workclass_sorted_te = workclass_list_te.most_common()
    b = 0
    for i in workclass_te:
        if('?' in i or 'NULL' in i):
            workclass_te[b] = workclass_sorted_te[0][0]
        b += 1

    #Native_country - Filling values with 'United-states'
    cntry_te = np.array(data_te['native_country'])
    country_list_te = collections.Counter(cntry_te)
    country_sorted_te = country_list_te.most_common()
    b = 0
    for i in country_te:
        if('?' in i or 'NULL' in i):
            country_te[b] = country_sorted_te[0][0]
        b += 1

        #Normalizing continuous data attributes in range zero to one
    #AGE
    age = [float(numeric_string) for numeric_string in data['age']]
    age_te = [float(numeric_string) for numeric_string in data_te['age']]
    max_age = max(max(age), max(age_te))
    min_age = min(min(age), min(age_te))
    diff = max_age - min_age
    for i in range (0, 520):
        age[i] = (age[i] - min_age) / (diff)
    for i in range (0, 288):
        age_te[i] = (age_te[i] - min_age) / (diff)

    #FNLWGT
    fnlwgt = [float(numeric_string) for numeric_string in data['fnlwgt']]
    fnlwgt_te = [float(numeric_string) for numeric_string in data_te['fnlwgt']]
    max_wt = max(max(fnlwgt), max(fnlwgt_te))
    min_wt = min(min(fnlwgt), min(fnlwgt_te))
    diff = max_wt - min_wt
    for i in range (0, 520):
        fnlwgt[i] = (fnlwgt[i] - min_wt) / (diff)
    for i in range (0, 288):
        fnlwgt_te[i] = (fnlwgt_te[i] - min_wt) / (diff)

    #Education Category - ordinal
    len_edu = len(set(edu))
    for i in range(0, 520):
        edu[i] = float((float(edu[i]) - 1) / len_edu)
    len_edu_te = len(set(edu_te))
    for i in range(0, 288):
        edu_te[i] = float((float(edu_te[i]) - 1) / len_edu)

    #Capital Gain
    gain = [float(numeric_string) for numeric_string in data['capital_gain']]
    gain_te = [float(numeric_string) for numeric_string in data_te['capital_gain']]
    max_gain = max(max(gain), max(gain_te))
    min_gain = min(min(gain), min(gain_te))
    diff = max_gain - min_gain
    for i in range (0, 520):
        gain[i] = (gain[i] - min_gain) / (diff)
    for i in range (0, 288):
        gain_te[i] = (gain_te[i] - min_gain) / (diff)

    #Capital Loss
    loss = [float(numeric_string) for numeric_string in data['capital_loss']]
    loss_te = [float(numeric_string) for numeric_string in data_te['capital_loss']]
    max_loss = max(max(loss), max(loss_te))
    min_loss = min(min(loss), min(loss_te))
    diff = max_loss - min_loss
    for i in range (0, 520):
        loss[i] = (loss[i] - min_loss) / (diff)
    for i in range (0, 288):
        loss_te[i] = (loss_te[i] - min_loss) / (diff)

    #Hours per Week
    hours_te = [float(numeric_string) for numeric_string in data_te['hour_per_week']]
    hours = [float(numeric_string) for numeric_string in data['hour_per_week']]
    max_hours = max(max(hours), max(hours_te))
    min_hours = min(min(hours), min(hours_te))
    diff = max_hours - min_hours
    for i in range (0, 520):
        hours[i] = (hours[i] - min_hours) / (diff)
    for i in range (0, 288):
        hours_te[i] = (hours_te[i] - min_hours) / (diff)



    #Putting all the attributes into a 2D array where each row represents a record in the dataset

    #one for continuous attributes - training dataset
    rec_cont = np.empty(shape=(520, 6))
    arr_cont = [age, fnlwgt, edu, gain, loss, hours]
    new_arr_cont = zip(*arr_cont)
    rec_cont = np.array(list(new_arr_cont))

    #one for nominal attributes - training dataset
    rec_nom = np.empty(shape=(520, 7))
    arr_nom = [workclass, m_status, occupation, rship, race, gender, country]
    new_arr_nom = zip(*arr_nom)
    rec_nom = np.array(list(new_arr_nom))

    #one for continuous attributes - test dataset
    rec_cont_te = np.empty(shape=(288, 6))
    arr_cont_te = [age_te, fnlwgt_te, edu_te, gain_te, loss_te, hours_te]
    new_arr_cont_te = zip(*arr_cont_te)
    rec_cont_te = np.array(list(new_arr_cont_te))

    #one for nominal attributes - test dataset
    rec_nom_te = np.empty(shape=(288, 7))
    arr_nom_te = [workclass_te, m_status_te, occupation_te, rship_te, race_te, gender_te, country_te]
    new_arr_nom_te = zip(*arr_nom_te)
    rec_nom_te = np.array(list(new_arr_nom_te))

    #Creating arrays for storing calculated proximities
    dist_euc = np.empty(shape=(288,520), dtype=float)
    dist_cos = np.empty(shape=(288,520), dtype=float)

    #implementing Eculidean Distance between rows of the data set
    def dist_eucl(x, y, a, b):
        cont_sum = 0
        for i in range (0, 6):
            sqr = (x[i]-y[i]) * (x[i]-y[i])
            cont_sum = cont_sum + sqr
        nom_sum = 0
        for i in range (0, 7):
            if (a[i] != b[i]):
                nom_sum += 1
        return math.sqrt(nom_sum + cont_sum)

    for i in range (0, 288):
        for j in range (0, 520):
                dist_euc[i][j] = dist_eucl(rec_cont_te[i], rec_cont[j], rec_nom_te[i], rec_nom[j])

    #implementing Cosine Similarity between rows of the data set
    def dist_cosine(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        return float(dot_product / (norm_a * norm_b))

    for i in range (0, 288):
        for j in range (0, 520):
            #cosine for continuous attributes
            cosine_similarity = dist_cosine(rec_cont_te[i], rec_cont[j])
            delta_cont = 1

            #workclass - nominal
            if (rec_nom_te[i][0] == rec_nom[j][0]):
                s_workclass = 1
                delta_workclass = 1
            else:
                s_workclass = 0
            delta_workclass = 1
            weight_workclass = s_workclass * delta_workclass

            #marital status - nominal
            if (rec_nom_te[i][1] == rec_nom[j][1]):
                s_marital = 1
            else:
                s_marital = 0
            delta_marital = 1
            weight_marital = s_marital * delta_marital

            #occupation - nominal
            if (rec_nom_te[i][2] == rec_nom[j][2]):
                s_occupation = 1
            else:
                s_occupation = 0
            delta_occupation = 1
            weight_occupation = s_occupation * delta_occupation

            #relationship - nominal
            if (rec_nom_te[i][3] == rec_nom[j][3]):
                s_rship = 1
            else:
                s_rship = 0
            delta_rship = 1
            weight_rship = s_rship * delta_rship

            #race - nominal
            if (rec_nom_te[i][4] == rec_nom[j][4]):
                s_race = 1
            else:
                s_race = 0
            delta_race = 1
            weight_race = s_race * delta_race

            #gender - nominal
            if (rec_nom_te[i][5] == rec_nom[j][5]):
                s_gender = 1
            else:
                s_gender = 0
            delta_gender = 1
            weight_gender = s_gender * delta_gender

            #native country - nominal
            if (rec_nom_te[i][6] == rec_nom[j][6]):
                s_country = 1
                delta_country = 1
            else:
                s_country = 0
            delta_country = 1
            weight_country = s_country * delta_country

            #calculating net weight of all the attributes divided by sum of their deltas give similarity and subtracting from 1 gives dissimilarity
            net_weight = cosine_similarity + weight_workclass + weight_marital + weight_occupation + weight_rship + weight_race + weight_gender + weight_country
            delta_total = delta_cont*6 + delta_workclass + delta_marital + delta_occupation + delta_rship + delta_race + delta_gender + delta_country
            dist_cos[i][j] = 1 - (net_weight / delta_total)


    #For confusion matrix
    posterior = np.empty([288], dtype=float)
    class_pred_str = np.empty(shape=(288), dtype=object)

    #Predicting class attribute for test dataset by assigning weihgt according to their distance

    #Eculidean
    class_pred_euc = np.empty([288], dtype=str)
    posterior_euc = np.empty([288], dtype=float)
    pred_class_euc = np.empty([288], dtype=str)
    list_final_euc = [[0 for j in range (4)] for i in range (288)]
    for i in range (0, 288):
        b = 0
        weight_class1 = 0
        weight_class2 = 0
        ind = dist_euc[i].argsort()
        list_final_euc[i][0] = str(i + 1)
        list_final_euc[i][1] = class_attr_te[i]
        for j in ind:
            if (b == k):
                break
            else:
                if '>50K' in class_attr[j]:
                    weight_class2 += (1/(dist_euc[i][j]**2))
                elif '<=50K' in class_attr[j]:
                    weight_class1 += (1/(dist_euc[i][j]**2))
            b += 1
        argmax = max(weight_class1, weight_class2)
        posterior_euc[i] = argmax / (weight_class1 + weight_class2)
        if(weight_class1 > weight_class2):
            pred_class_euc[i] = '<=50K'
            list_final_euc[i][2] = '<=50K'
            class_pred_euc[i] = '<=50K'
        else:
            pred_class_euc[i] = '>50K'
            list_final_euc[i][2] = '>50K'
            class_pred_euc[i] = '>50K'
        list_final_euc[i][3] = str(posterior_euc[i])

    #cosine
    class_pred_cos = np.empty([288], dtype=str)
    posterior_cos = np.empty([288], dtype=float)
    pred_class_cos = np.empty([288], dtype=str)
    list_final_cos = [[0 for j in range (4)] for i in range (288)]
    for i in range (0, 288):
        b = 0
        count_class1 = float(0)
        count_class2 = float(0)
        weight_class1 = 0
        weight_class2 = 0
        ind = dist_cos[i].argsort()
        list_final_cos[i][0] = str(i + 1)
        list_final_cos[i][1] = class_attr_te[i]
        for j in ind:
            if (b == k):
                break
            else:
                if '>50K' in class_attr[j]:
                    count_class2 += 1
                    weight_class2 += (1/(dist_cos[i][j]**3))
                elif '<=50K' in class_attr[j]:
                    count_class1 += 1
                    weight_class1 += (1/(dist_cos[i][j]**3))
            b += 1
        argmax = max(weight_class1, weight_class2)
        posterior_cos[i] = argmax / (weight_class1 + weight_class2)
        if(weight_class1 > weight_class2):
            pred_class_cos[i] = '<=50K'
            list_final_cos[i][2] = '<=50K'
            class_pred_cos[i] = '<=50K'
        else:
            pred_class_cos[i] = '>50K'
            list_final_cos[i][2] = '>50K'
            class_pred_cos[i] = '>50K'
        list_final_cos[i][3] = str(posterior_cos[i])

    #preparing output for writing into a csv file
    heading_list = []
    heading_list.append('Record Number')
    heading_list.append('Actual Class')
    heading_list.append('Predicted Class')
    heading_list.append('Pesterior Probability')

    #writing to csv file
    with open('output_homework2_joshi_euc.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(list_final_euc)

    with open('output_homework2_joshi_cosine.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(list_final_euc)


