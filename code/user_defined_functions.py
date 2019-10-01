import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# ~~ Create frequency plot of transactions by fraud type ~~ #
def make_frequency_plot(iv):
    # Initialize figure 
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    # Sum fraud/non-fraud cases
    y_values=[sum(iv==0),sum(iv==1)]
    x_range=[0,0.25]

    # Create bar plot
    plt.bar(x_range,
            y_values,
            width=(0.50)*(x_range[1]-x_range[0]),
           alpha=0.65)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set axis tick values
    ax.tick_params(axis=u'x', which=u'both',length=0)
    plt.yticks(fontsize=14,fontname="Serif")
    plt.xticks(x_range,
               ['Non-Fraudulent',"Fraudulent"],
               fontsize=18,
               fontname='Serif')

    # Set axis labels #
    plt.ylabel('Number of Transactions',
               fontsize=16,
               fontname='Serif')

    # Create Title #
    plt.title('Number of Transactions \n by Fraud Status',
              fontsize=18,
              fontname='Serif',
              y=1.08)

    # Add labels above bar plot
    ax.text(x_range[0]-0.07, y_values[0]+10000, '{0:,.0f} Cases'.format(y_values[0]), 
            color='black', 
            fontweight='bold',
            fontsize=18)
    ax.text(x_range[1]-0.05, y_values[1]+10000, '{0:<} Cases'.format(y_values[1]), 
            color='red', 
            fontweight='bold',
            fontsize=18)

    # Show plot
    plt.show()


class oversample_KFold(KFold):
    '''Extends KFold class for binary datasets to allow for 
        stratified sampling (or over-sampling) of the positive
        class
        
       Methods:
           yield_sample (list) : yields train/test indices after stratified 
                                 sampling using sample_weight as the probability
                                 of choosing observations from the positive class

    '''
    
    # ~ Initialize ~ #
    def __init__(self,*args, **kwargs):
        '''Add sample_weights keyword argument to KFold class'''
        
        self.sample_weight=kwargs.pop('sample_weight')
        super(oversample_KFold, self).__init__(*args, **kwargs)
        
    # ~ Extend KFold class to allow oversampling ~ #
    def yield_sample(self,df):
        '''Run KFold split, within the training dataset use a stratified sampling and yield
           test and train indices using this sampling procedure.
           
           Inputs:
             df (dataframe) : data used in model estimation
             
           Outputs:
             stratified_train_ix, test_ix (tuple) : tuple of training and test indices (same as 
                                           KFold class)
           
           Note: 
           
            To form the stratified_train_ix we use the following sampling procedure goes as follows.
            Create a sample of size df.shape[0]. Within each row, we draw a pair of random numbers:
            
            1. Draw Z from a uniform random. 
               If Z <= sample_weight : in step (2) sample from the negative class (class = 0)
               If Z >  sample_weight : in step (2) sample from the positivie class (class = 1)
            
            2. Draw W from a discrete uniform distribution with size = number of observations in
                the class being drawn from.
               
               For example, if:
                              sample weight=0.5
                              Z = 0.3
                              Number Obs in Class 0 = 10
                              Number Obs in Class 1 = 20
                            then: 
                              W is drawn from discrete uniform with size = 10 (i.e. each observation
                                has a 1/10 probability of being drawn)
               
          These pairs (Z,W) create a new set of indices for our test dataset. 
        
        '''
        # ~ Indices for 0 and 1 class ~ #
        class_0_ix=set(df.loc[df['Class']==0].index.values)
        class_1_ix=set(df.loc[df['Class']==1].index.values)
        for split in KFold(n_splits=self.n_splits,random_state=self.random_state).split(df):
            # ~ Get train/test indices ~ #
            train_ix = split[0]
            test_ix = split[1]
            # ~ Split training data into class 0 and 1 training ~ #
            class_0_train = [x for x in train_ix if x in class_0_ix]
            class_1_train = [x for x in train_ix if x in class_1_ix]
            # ~ Draw a random variable to determine which class to sample ~ #
            select_class_draws = np.random.uniform(size=len(train_ix))<=self.sample_weight
            n_class_0 = sum(select_class_draws==0)
            n_class_1 = sum(select_class_draws==1)
            # ~ Within each class, draw a random number to select each index ~ #
            oversample_train_ix = ([class_0_train[x] for x in np.random.randint(low=0,high=len(class_0_train)-1,size=n_class_0)] +
                                     [class_1_train[x] for x in np.random.randint(low=0,high=len(class_1_train)-1,size=n_class_1)])
            # ~ Yield indices for train and test data ~ #
            stratified_train_ix=np.array(oversample_train_ix)
            yield stratified_train_ix, test_ix