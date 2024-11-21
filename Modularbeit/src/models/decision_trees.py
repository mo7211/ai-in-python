from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils._train import define_target

def train_decision_tree(df:DataFrame):
    #generate the classification object
    decisiontree = DecisionTreeClassifier(random_state=0)

    target, features = define_target(df, 'condition')

    #train the model
    model = decisiontree.fit(features, target)

    df


    #make a new observation (randomly)
    observation = [[ 5,  4,  3,  2]]

    #predict the class of the observation
    model.predict(observation)
