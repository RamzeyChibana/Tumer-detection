from utils.DataPre import get_regions , get_features
import numpy as np
import os 
from PIL import Image
from utils.parser import data_parse
from tqdm import tqdm

def save_fetures(path,new_path,rows,columns,num_bins):
    categories = os.listdir(path)
    for category in categories:
        os.makedirs(os.path.join(new_path,f"{rows}_{columns}_{num_bins}",category))
    categories_map = {category:i for i,category in enumerate(categories)}
    files = []
    labels = []
    
    for category in categories :
        for file in os.listdir(os.path.join(path,category)):
            image = np.array(Image.open(os.path.join(path,category,file)).convert("L"))
            regions = get_regions(image,rows,columns)
            features = get_features(regions,num_bins)
            np.save(os.path.join(new_path,f"{rows}_{columns}_{num_bins}",category,file.split(".")[0]),features) 
            
    
    



if __name__ == "__main__":
    parser = data_parse()
    args = parser.parse_args()
    rows = args.rows
    columns = args.columns
    num_bins = args.bins
    print(f"features with :{rows},{columns},{num_bins}")
    save_fetures("D:\df\\ai\Clean Data\clean_images","D:\df\\ai\Clean Data\clean_features",rows,columns,num_bins)
    
    






