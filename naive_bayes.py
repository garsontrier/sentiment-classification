import os
import re
import numpy as np
import string
import random

stop_words = [
'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'a', 'and', 'any', 'are', 'arent', 'as', 'at', 'be', 'because',
'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could', 'couldnt', 'did', 'didnt',
'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadnt',
'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him',
'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself',
'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
'ought', 'our', 'ours',	'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should',
'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under',
'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where',
'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd',
'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves', '']


def process_sentences(path, by_word):  # preprocess training or test set, removing punctuation and stop words
    trans = str.maketrans('', '', string.punctuation)
    file_list = os.listdir(path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    words = []
    for i in file_list:
        with open(path + i, 'r') as f:
            file = re.split(r'[\s]+', f.read().translate(trans))
        for word in file:
            if word in stop_words:
                file.remove(word)
        if by_word:
            words.extend(file)
        else:
            words.append(file)
    return words


def remove_rep_for_all(file):
    no_rep = []
    length = 0
    for i in range(len(file)):
        no_rep.append([])
        for j in file[i]:
            length = length + 1
            if j not in no_rep[i]:
                no_rep[i].append(j)
    return no_rep, length


def word_probabilities_mn(path="./data/train/"):  # computes word probabilites for Multinomial NB and saves it
    # enter pos folder
    if not (os.path.isfile('./pos_prob_mn.npy') and os.path.isfile('./neg_prob_mn.npy') and os.path.isfile('./vocabulary.txt')):
        pos = []
        pos_prob_list = []
        pos_words = process_sentences(path+"pos/", True)
        for i in pos_words:
            if i not in pos:
                pos.append(i)
        neg_words = process_sentences(path+"neg/", True)
        neg = []
        neg_prob_list = []
        for i in neg_words:
            if i not in neg:
                neg.append(i)
        voc = []
        voc.extend(pos)
        for i in neg:
            if i not in voc:
                voc.append(i)
        for i in voc: # compute word probs by counting each words occurrence in training set
            nump = pos_words.count(i)
            numn = neg_words.count(i)
            pos_prob_list.append(nump+1)
            neg_prob_list.append(numn+1)
        pos_prob = np.asarray(pos_prob_list, dtype=float)
        neg_prob = np.asarray(neg_prob_list, dtype=float)
        pos_prob = np.log(pos_prob / (len(pos_words) + len(voc)))
        neg_prob = np.log(neg_prob/(len(neg_words) + len(voc)))
        with open('./vocabulary.txt', 'w') as f:
            for i in voc:
                f.write('%s\n' % i)
        np.save("./pos_prob_mn", pos_prob)
        np.save("./neg_prob_mn", neg_prob)
    else:
        pos_prob = np.load('./pos_prob_mn.npy')
        neg_prob = np.load('./neg_prob_mn.npy')
        with open('./vocabulary.txt', 'r') as f:
            voc = f.read().split('\n')
    return pos_prob, neg_prob, voc


def word_probabilities_ber(path="./data/train/"): # computes word probabilites for Bernoulli NB and saves it
    if not (os.path.isfile('./pos_prob_ber.npy') and os.path.isfile('./neg_prob_ber.npy')):
        pos_words_file = process_sentences(path+"pos/", False)
        neg_words_file = process_sentences(path+"neg/", False)
        pos_prob_list = []
        neg_prob_list = []
        _, _, voc = word_probabilities_mn()
        for i in voc: # compute word probs by counting number of files that a certain word appears
            nump = 1
            numn = 1
            for j in pos_words_file:
                if i in j:
                    nump = nump + 1
            for j in neg_words_file:
                if i in j:
                    numn = numn + 1
            pos_prob_list.append(nump)
            neg_prob_list.append(numn)
        pos_prob = np.asarray(pos_prob_list, dtype=float)
        neg_prob = np.asarray(neg_prob_list, dtype=float)
        pos_prob = pos_prob/(len(pos_words_file)+2)  # add-1 smoothing
        neg_prob = neg_prob/(len(neg_words_file)+2)
        np.save('./pos_prob_ber', pos_prob)
        np.save('./neg_prob_ber', neg_prob)
    else:
        pos_prob = np.load('./pos_prob_ber.npy')
        neg_prob = np.load('./neg_prob_ber.npy')
    return pos_prob, neg_prob


def word_probabilities_bin(path='./data/train/'):  # computes word probabilites for Binary NB and saves it
    if not (os.path.isfile('./pos_prob_bin.npy') and os.path.isfile('./neg_prob_bin.npy')):
        pos_file = process_sentences(path+'pos/', False)
        neg_file = process_sentences(path+'neg/', False)
        _, _, voc = word_probabilities_mn()
        pos_no_rep, pos_len = remove_rep_for_all(pos_file)
        neg_no_rep, neg_len = remove_rep_for_all(neg_file)
        pos = []
        neg = []
        pos_prob_list = []
        neg_prob_list = []
        for i in pos_no_rep:
            pos.extend(i)
        for i in neg_no_rep:
            neg.extend(i)
        for i in voc: # compute word probs by counting number of occurence of a word in the training set w/o repetition
            nump = pos.count(i)
            numn = neg.count(i)
            pos_prob_list.append(nump+1)
            neg_prob_list.append(numn+1)
        pos_prob = np.log(np.asarray(pos_prob_list)/(len(voc) + pos_len))
        neg_prob = np.log(np.asarray(neg_prob_list)/(len(voc) + neg_len))
        np.save('./pos_prob_bin.npy', pos_prob)
        np.save('./neg_prob_bin.npy', neg_prob)
    else:
        pos_prob = np.load('./pos_prob_bin.npy')
        neg_prob = np.load('./neg_prob_bin.npy')
    return pos_prob, neg_prob


def test(type, path='./data/test/'): # creates output of the classifier on the test set given the classifier type
    pos_files = process_sentences(path+"pos/", False)
    neg_files = process_sentences(path+"neg/", False)
    files = []
    files.extend(pos_files)
    files.extend(neg_files)
    true_key = np.concatenate((np.zeros([1, len(pos_files)]), np.ones([1, len(neg_files)])), axis=1)
    key = []
    class_prob = [np.log(1 / 2), np.log(1 / 2)]
    if type == 'mn':
        pos_prob, neg_prob, voc = word_probabilities_mn()
    elif type == 'bin':
        _, _, voc = word_probabilities_mn()
        pos_prob, neg_prob = word_probabilities_bin()
        files, _  = remove_rep_for_all(files)
    else:
        _, _, voc = word_probabilities_mn()
        pos_prob, neg_prob = word_probabilities_ber()
        for i in files:
            sump = class_prob[0]
            sumn = class_prob[1]
            for j in range(len(voc)):
                if voc[j] in i:
                    sump = sump + np.log(pos_prob[j])
                    sumn = sumn + np.log(neg_prob[j])
                else:
                    sump = sump + np.log(1-pos_prob[j])
                    sumn = sumn + np.log(1-neg_prob[j])
            if sump > sumn:
                key.append(0)
            else:
                key.append(1)
        return key, true_key

    for i in files:
        sump = class_prob[0]
        sumn = class_prob[1]
        for j in i:
            try:
                ind = voc.index(j)
            except ValueError:
                ind = -100
            if ind != -100:
                sump = sump + pos_prob[ind]
                sumn = sumn + neg_prob[ind]
        if sump >= sumn:
            key.append(0)
        else:
            key.append(1)
    return np.asarray(key), true_key


def evaluate(key, true_key, type):  # prints out performance metrics given the type of classifier
    pos, neg = contingency_table(key, true_key)
    precision_pos, recall_pos, f_measure_pos = precision_recall_f_measure(pos)
    precision_neg, recall_neg, f_measure_neg = precision_recall_f_measure(neg)
    macro_averaged_precision = precision_neg/2 + precision_pos/2
    macro_averaged_recall = recall_pos/2 + recall_neg/2
    macro_averaged_f_measure = f_measure_pos/2 + f_measure_neg/2
    micro_averaged_precision, micro_averaged_recall, micro_averaged_f_measure = micro_averaged_results(pos, neg)
    if type == 'mn':
        name = 'Multinomial Naive Bayes'
    elif type == 'ber':
        name = 'Multivariate Naive Bayes'
    else:
        name = 'Binary Naive Bayes'
    print()
    print('Results for ' + name + ' system')
    print('Results of Positive Class')
    print('Precision: ' + str(precision_pos))
    print('Recall: ' + str(recall_pos))
    print('F-measure: ' + str(f_measure_pos))
    print('Results for Negative Class')
    print('Precision: ' + str(precision_neg))
    print('Recall: ' + str(recall_neg))
    print('F-measure: ' + str(f_measure_neg))
    print('Micro-Averaged Results')
    print('Precision: ' + str(micro_averaged_precision))
    print('Recall: ' + str(micro_averaged_recall))
    print('F-measure: ' + str(micro_averaged_f_measure))
    print('Macro-Averaged Results')
    print('Precision: ' + str(macro_averaged_precision))
    print('Recall: ' + str(macro_averaged_recall))
    print('F-measure: ' + str(macro_averaged_f_measure))
    print('------------------------------')


def contingency_table(key,true_key):  # 0 - tp, 1 - fp, 2 - fn, 3 - tn
    sum = key + 2 * true_key
    pos = []
    pos.append(np.count_nonzero(sum == 0))  # tp
    pos.append(np.count_nonzero(sum == 2))  # fp
    pos.append(np.count_nonzero(sum == 1))  # fn
    pos.append(np.count_nonzero(sum == 3))  # tn
    pos.reverse()
    neg = pos.copy()
    pos.reverse()
    return pos, neg


def precision_recall_f_measure(cont):
    recall = cont[0] / (cont[0] + cont[2])
    precision = cont[0] / (cont[0] + cont[1])
    f_meas = f_measure(precision, recall)
    return precision, recall, f_meas


def f_measure(precision, recall):
    return 2*precision*recall/(precision + recall)


def micro_averaged_results(pos, neg):
    con = [x + y for x, y in zip(pos, neg)]
    pre, rec, f = precision_recall_f_measure(con)
    return pre, rec, f


def micro_averaged_f_measure(key1, key2, true_key):
    pos1, neg1 = contingency_table(key1, true_key)
    pos2, neg2 = contingency_table(key2, true_key)
    con1 = [x + y for x, y in zip(pos1, neg1)]
    con2 = [x + y for x, y in zip(pos2, neg2)]
    _, _, fa = precision_recall_f_measure(con1)
    _, _, fb = precision_recall_f_measure(con2)
    return fa, fb


def randomization_test(key1, key2, true_key, p=0.05, R=1000): # performs randomization test with micro-averaged f-measures given two systems' outputs
    counter = 0
    fa, fb = micro_averaged_f_measure(key1, key2, true_key)
    s = abs(fa-fb)
    key1_r = np.zeros_like(key1)
    key2_r = np.zeros_like(key2)
    for i in range(R):
        for j in range(len(key1)):
            a = random.randint(0, 1)
            if a == 0:  # don't shuffle
                key1_r[j] = key1[j]
                key2_r[j] = key2[j]
            else:
                key1_r[j] = key2[j]
                key2_r[j] = key1[j]
        fa_r, fb_r = micro_averaged_f_measure(key1_r, key2_r, true_key)
        s_r = abs(fa_r - fb_r)
        if s_r >= s:
            counter = counter + 1
    p_r = (counter+1)/(R+1)
    if p_r <= p:
        print('Those two systems are different according to the randomization test with p-value = ' + str(p))
    else:
        print('Those two systems are not different according to the randomization test with p-value = ' + str(p))
        print('Minimum p-value that this systems are identified as different systems is: ' + str(p_r))


if __name__ == "__main__":
    naive_types = ['mn', 'ber', 'bin']
    naive_tuples = [[0, 1], [0, 2], [1, 2]]
    key = np.zeros([3, 600])
    for i in range(len(naive_types)): # compute output of each classifier
        key[i], true_key = test(type=naive_types[i])
    for i in naive_tuples: # perform randomization test for all combinations of classifiers
        print('Randomization test for ' + naive_types[i[0]] + ' and ' + naive_types[i[1]])
        randomization_test(key[i[0]], key[i[1]], true_key)
    for i in range(len(naive_types)): # print out performance metrics for all classifiers
        evaluate(key[i], true_key, naive_types[i])



