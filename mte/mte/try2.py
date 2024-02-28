import pandas as pd
import matplotlib.pyplot as plt
data="https://drive.google.com/uc?export=download&id=1gSjYHJ8OPM9HMd3prr7XuhvSWWGKYZNE"
lefthanded_data= pd.read_csv(data)
# Create a scatter plot of "Male" and "Female" columns vs. "Age"
lefthanded_data.plot(x='Age', y=['Male', 'Female'], kind='scatter')
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.title("Left-Handedness by Age")
plt.show()