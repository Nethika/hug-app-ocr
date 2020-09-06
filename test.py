import string
import numpy as np

def find_score(info_type,text):

    if info_type == 'expire':
        #print('/ + numbers!')

        chars = [s for s in text]

        if "/" in chars:
            s_1 = 0.8
        else:
            s_1 = 0.1

        rest = set(chars) - set(["/"])
        score_ex =[]
        for ch in rest:
            if ch in string.digits:
                s_2 = 0.8
            else:
                s_2 = 0.1
            score_ex.append(s_2)

        score_ex.append(s_1)

        if len(text) == 5:
            s_3 = 0.8
        else:
            s_3 = 0.01
        score_ex.append(s_3)

        score = np.mean(score_ex)
        print("score:",score)


    if info_type == 'name':
        #print('everything should be chars!')
        chars = [s for s in text]
        score_na =[]
        for ch in chars:
            if ch.isalpha():
                s_3 = 0.8
            elif ch == " ":
                s_3 = 0.5
            else:
                s_3 = 0.1
            score_na.append(s_3)

        score = np.mean(score_na)

                
    if info_type == 'number':
        #print('everything should be numbers!')    

        chars = [s for s in text]
        score_num =[]
        for ch in chars: 
            if ch in string.digits:
                s_1 = 0.8
            elif ch == " ":
                s_1 = 0.8
            else:
                s_1 = 0.1
            score_num.append(s_1)
        if len(text) == 19:
            s_2 = 0.8
        else:
            s_2 = 0.2
        score_num.append(s_2)
        score = np.mean(score_num)
        print("score:",score)

    return score

info_type = 'number'
text = "1111 000 2342 2345"

info_type = 'expire'
text = "01/2366"

score = find_score(info_type,text)