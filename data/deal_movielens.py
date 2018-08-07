import re, sys, os, random, time
import numpy as np
def read_ratings(filename, mlKBflag):
    f = open(filename)
    #UserID::MovieID::Rating::Timestamp
    user, All_ratings, item_time, item_rating= "null", {}, {}, {} #the All_ratings is {user : [[items], [scores]]}
    f.readline()
    while 1:
        line = f.readline()
        if not line:
            break
        temp_str = re.split(":|\n|\r|\t|,", line)
        #print temp_str
	ss = []
        for i in range(1, len(temp_str)):
            if temp_str[i] != "":
                ss.append(temp_str[i])
        mid, rating, timestamp = "m"+ss[0], ss[1], int(ss[2])
	if user == "null":
	    user = temp_str[0]
        if user != temp_str[0]:
            t = sorted(item_time.iteritems(), key=lambda d:d[1])
	    #random.shuffle(t)
            item_time, scores, times = [], [], []#the All_ratings is {user : [[items], [scores], [times]]}
            for (item, Time) in t:
                item_time.append(item)
                scores.append(item_rating[item])
		times.append(Time)
	    t = len(item_time)
	    if t >= 10 and t <= 5000:
                #All_ratings[user] = [item_time, scores, times]
		All_ratings[user] = [item_time[-200:], scores[-200:], times[-200:]]
            # the next user
            user = temp_str[0]
            item_time, item_ranting = {}, {}
	if mlKBflag:
            item_time[mid] = timestamp # user : {mid : time}
            item_rating[mid] = rating # user : {mid : rating}
    # the final user
    t = sorted(item_time.iteritems(), key=lambda d:d[1])
    #random.shuffle(t)
    items, scores, times = [], [], [] #the All_ratings is {user : [[items], [scores], [times]]}
    for (item, Time) in t:
        items.append(item)
        scores.append(item_rating[item])
	times.append(Time)
    t = len(item_time)
    if t >= 10 and t <= 5000:
        All_ratings[user] = [items[-200:], scores[-200:], times[-200:]]    
	#All_ratings[user] = [items, scores, times]
    f.close()
    return All_ratings #All_ratings is {user : [[items], [scores]]}

def PrintBPR(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
        for i in range(len(items)):
            fp.write(user + " " + items[i][1:] + " " + "1" + "\n")
    fp.close()

def PrintNCF(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        for i in range(len(items)-1, -1, -1):
            #fp.write(user + "\t" + items[i][1:] + "\t" + str(scores[i]) + "\t" + str(times[i]) + "\n") #get items[i][1:], because the mid consist of "m" and ID
	    fp.write(user + "\t" + items[i][1:] + "\t" + str(1.0) + "\t" + str(times[i]) + "\n") #get items[i][1:], because the mid consist of "m" and ID
    fp.close()

def PrintSequence(All_ratings, filename):
    fp = open(filename, 'w')
    for user in All_ratings:
        items, scores = All_ratings[user][0], All_ratings[user][1]
	fp.write(user)
        for i in range(len(items)):
	    #if scores[i] < 3:
	    #   break
            fp.write(" " + items[i])
	fp.write("\n")
    fp.close()

def cut_train_test_set(All_ratings, ratio): #ratio is float
    train_ratings, test_ratings = {}, {}
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        train_num = int(round(len(items) * ratio))
        train_items, train_scores, train_times = items[:train_num], scores[:train_num], times[:train_num]
        test_items, test_scores, test_times = items[train_num:], scores[train_num:], times[train_num:]
        train_ratings[user] = [train_items, train_scores, train_times]
        test_ratings[user] = [test_items, test_scores, test_times]
    #PrintLINE(train_ratings, sys.argv[1] + "train" + str(ratio) + "_LINE")
    #PrintLINE(test_ratings, sys.argv[1] + "test" + str(ratio) + "_LINE")
    PrintSequence(train_ratings, sys.argv[1] + "_train." + str(ratio))
    PrintSequence(test_ratings, sys.argv[1] + "_test." + str(ratio))
    PrintNCF(train_ratings, sys.argv[1] + "train_NCF." + str(ratio))
    PrintNCF_transe(test_ratings, sys.argv[1] + "test_NCF." + str(ratio))
    #PrintNCF_transe(test_ratings, sys.argv[1] + "test_NCF." + str(ratio))
    PrintBPR(train_ratings, sys.argv[1] + "train" + str(ratio) + "_BPR")

def cut_train_test_set_one_out(All_ratings): 
    train_ratings, test_ratings = {}, {}
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        train_num = len(items) - 1
        train_items, train_scores, train_times = items[:train_num], scores[:train_num], times[:train_num]
        test_items, test_scores, test_times = items[train_num:], scores[train_num:], times[train_num:]
        train_ratings[user] = [train_items, train_scores, train_times]
        test_ratings[user] = [test_items, test_scores, test_times]
    
    #PrintLINE(train_ratings, sys.argv[1] + "train" + str(ratio) + "_LINE")
    #PrintLINE(test_ratings, sys.argv[1] + "test" + str(ratio) + "_LINE")
    #PrintSequence(test_ratings, sys.argv[1] + "_test.oneout")
    #PrintSequence(train_ratings, sys.argv[1] + "_train.oneout")
    #PrintWord2vec(train_ratings, sys.argv[1] + "_train_w2v.oneout")
    PrintBPR(train_ratings, sys.argv[1] + "train_BPR.oneout")
    PrintNCF(train_ratings, sys.argv[1] + "train_NCF.oneout")
    PrintNCF(test_ratings, sys.argv[1] + "test_NCF.oneout")

def get_last_day(last_time):
    t = time.localtime(last_time)
    last_day = time.struct_time((t[0],t[1],t[2],0,0,0,0,0,0))
    last_day = time.mktime(last_day)
    return last_day
def get_last_index(all_times):
    last_day = get_last_day(all_times[-1])
    last_index = 0
    for i in range(len(all_times)-1, -1,-1):
        if all_times[i] < last_day:
            last_index = i+1
            break
    return last_index
def cut_train_test_set_oneSess_out(All_ratings):
    train_ratings, test_ratings = {}, {}
    for user in All_ratings:
        items, scores, times = All_ratings[user][0], All_ratings[user][1], All_ratings[user][2]
        last_index = get_last_index(times)
        train_num = last_index
	if train_num < 5 or (2 * train_num) < len(items):
            continue
        train_items, train_scores, train_times = items[:train_num], scores[:train_num], times[:train_num]
        test_items, test_scores, test_times = items[train_num:], scores[train_num:], times[train_num:]
        train_ratings[user] = [train_items, train_scores, train_times]
        test_ratings[user] = [test_items, test_scores, test_times]
    #PrintLINE(train_ratings, sys.argv[1] + "train" + str(ratio) + "_LINE")
    #PrintLINE(test_ratings, sys.argv[1] + "test" + str(ratio) + "_LINE")
    PrintSequence(test_ratings, sys.argv[1] + "_test.oneSessout_check")
    #PrintSequence(train_ratings, sys.argv[1] + "_train.oneSessout")
    #PrintWord2vec(train_ratings, sys.argv[1] + "_train_w2v.oneSessout")
    #PrintBPR(train_ratings, sys.argv[1] + "train_BPR.oneSessout")
    #PrintNCF(train_ratings, sys.argv[1] + "train_NCF.oneSessout")
    #PrintNCF(test_ratings, sys.argv[1] + "test_NCF.oneSessout")

if __name__ == "__main__":
      
    ml2KBflag = True
    #print sum(ml2KBflag)
    All_ratings = read_ratings(sys.argv[1], ml2KBflag)
    
    cut_train_test_set_one_out(All_ratings)
    

