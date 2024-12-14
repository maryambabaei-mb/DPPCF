
import numpy as np

#get the counterfactual as input
# consider having some information about the data (I am not sure which information we will have)
# search for instances in the Training DS which look like to the input instance
# return the number (or percentage) of data which share this information with the counterfactual
def reidentify(instance,dataset):
    count = 0
    for i in range(len(dataset)):
        a = np.where(instance == dataset[i])[1]
        if len(a) == len(instance[0]) :
            count +=1

    return count



# def unique_instances(dataset,labels):
#     uniqu_set = []
#     tmp_dataset = dataset.copy()
#     for instance in tmp_dataset:
#         uniqu_set.append(instance)
#         tmp_dataset,tmp_labels = remove_similar(tmp_dataset,labels,instance)
    
    
#     return uniqu_set


def unique_instances(dataset,labels):
    
    mask = np.ones(len(dataset), dtype=bool)
    similar_count = []
    for instance in dataset:
        count = 0
        first = True
        for i in range(len(dataset)):
            if(mask[i]):
                a = np.where(instance.reshape(1,-1) == dataset[i])[1]
                if len(a) == len(instance) and not first:  # if it is the first instance with these feature values it should not be removed
                    mask[i] = False
                    count +=1
                elif len(a) == len(instance) and first:
                    first = False
        similar_count.append(count)
    
    dataset_filtered = dataset[mask]
    labels_filtered = labels[mask]

    return dataset_filtered,labels_filtered,similar_count
