from sklearn.ensemble import RandomForestClassifier
import numpy as np

trainlist = []
testlist = []

def lettersEntropy(name):
    letters=[i for i in name if i.isalpha()]
    l,counts=np.unique(letters,return_counts=True)
    all=sum(counts)
    pro=list(map(lambda x:x/all,counts))
    entropy=sum(-n*np.log(n) for n in pro)
    return entropy

def featureInit(name):
    number=sum(i.isdigit() for i in name)
    token=name.split(".")
    segment=len(token[0])
    return [len(name),number,lettersEntropy(name),segment]

class Domain:
    def __init__(self,_name,_label):
        self.name = _name
        self.label = _label
        
    def returnData(self):
        return featureInit(self.name)

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1
        
def loadTrainData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            trainlist.append(Domain(name,label))

def loadTestData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            testlist.append(line)

def main():
    loadTrainData("train.txt")
    loadTestData("test.txt")
    trainFeature = []
    trainLabel = []
    for item in trainlist:
        trainFeature.append(item.returnData())
        trainLabel.append(item.returnLabel())
    
    clf = RandomForestClassifier(random_state=0)
    clf.fit(trainFeature,trainLabel)
    testFeature=[featureInit(i) for i in testlist]
    predictList=clf.predict(testFeature)
    
    with open("result.txt","w") as f:
        for i in range(len(predictList)):
            if predictList[i]==0:
                f.write(testlist[i]+",notdga\n")
            else:
                f.write(testlist[i]+",dga\n")
    
if __name__ == '__main__':
    main()