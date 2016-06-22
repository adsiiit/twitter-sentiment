from statistics import mode
import pickle

data_f = open("pickled/data.pickle", "rb")
data = pickle.load(data_f)
data_f.close()

vect_f = open("pickled/vect.pickle", "rb")
vect = pickle.load(vect_f)
vect_f.close()

mnb_f = open("pickled/mnb.pickle", "rb")
mnb = pickle.load(mnb_f)
mnb_f.close()

knn_f = open("pickled/knn.pickle", "rb")
knn = pickle.load(knn_f)
knn_f.close()

bnb_f = open("pickled/bnb.pickle", "rb")
bnb = pickle.load(bnb_f)
bnb_f.close()

sgdc_f = open("pickled/sgdc.pickle", "rb")
sgdc = pickle.load(sgdc_f)
sgdc_f.close()

nsvc_f = open("pickled/nsvc.pickle", "rb")
nsvc = pickle.load(nsvc_f)
nsvc_f.close()

lsvc_f = open("pickled/lsvc.pickle", "rb")
lsvc = pickle.load(lsvc_f)
lsvc_f.close()

logreg_f = open("pickled/logreg.pickle", "rb")
logreg = pickle.load(logreg_f)
logreg_f.close()



def final(text):
    inarray = []
    inarray.append(text)
    output = vect.transform(inarray)
    result = []
    result.append(lsvc.predict(output)[0])
    result.append(mnb.predict(output)[0])
    result.append(knn.predict(output)[0])
    result.append(bnb.predict(output)[0])
    result.append(sgdc.predict(output)[0])
    result.append(nsvc.predict(output)[0])
    result.append(logreg.predict(output)[0])
    
    finalresult = mode(result)
    
    choice_votes = result.count(mode(result))
    conf = choice_votes / float(len(result))
    confper = conf*100

    
    return finalresult, confper
