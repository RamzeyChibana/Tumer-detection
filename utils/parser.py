import argparse




def train_parse():
    parse = argparse.ArgumentParser(description="Train Model")
    parse.add_argument('-bs',"--batch_size",type=int,default=16,help="batch size")
    parse.add_argument('-dv',"--device",default="gpu",help="device")
    parse.add_argument('-v',"--verbose",default=2,type=int,help="device")
    parse.add_argument('-ep',"--epochs",type=int,default=10,help="how many epochs ")
    parse.add_argument('-dp',"--dropout",type=int,default=0.4,help="dropout rate")
    parse.add_argument('-lr',"--learning_rate",type=float,default=1e-3,help="learning rate")
    parse.add_argument("-ly","--layers",nargs="+",type=int,default=[64,64,128],help="layers size")
    parse.add_argument("-ex","--exp",type=int,default=None,help="start new training or continue experiment")
    parse.add_argument("-m","--model",default="MLP",choices=["MLP","CNN"])
    parse.add_argument("-dt","--dataset",default="freq",choices=["freq","images"])
    parse.add_argument('-r',"--rows",type=int,default=3,help="number of rows")
    parse.add_argument('-c',"--columns",type=int,default=3,help="number of columns")
    parse.add_argument('-b',"--bins",default=30,type=int,help="number of beans")
    return parse


def test_parse():
    parse = argparse.ArgumentParser(description="Test experiment")
    parse.add_argument('-dv',"--device",default="gpu",help="device")
    parse.add_argument('-b',"--batch_size",type=int,default=16,help="batch size")
    parse.add_argument("-ex","--exp",type=int,default=None,help="start new training or continue experiment")
    return parse


 
def plot_hist():
    parse = argparse.ArgumentParser(description="Plot history")
    parse.add_argument("-ex","--exp",type=int,default=None,help="plot exp history")
    return parse



def data_parse():
    parse = argparse.ArgumentParser(description="change Data format")
    parse.add_argument('-r',"--rows",type=int,default=3,help="number of rows")
    parse.add_argument('-c',"--columns",type=int,default=3,help="number of columns")
    parse.add_argument('-b',"--bins",default="10",type=int,help="number of beans")
    return parse

