import pandas as pd
import matplotlib.pyplot as plt

# Load the handedness data
data_url_1 = "https://gist.githubusercontent.com/mbonsma/8da0990b71ba9a09f7de395574e54df1/raw/aec88b30af87fad8d45da7e774223f91dad09e88/lh_data.csv"

lefthanded_data = pd.read_csv(data_url_1)

# Creating a scatter plot of "Male" and "Female" columns vs. "Age"
lefthanded_data.plot(x='Age', y=['Male', 'Female'])
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.title("Left-Handedness by Age")
plt.show()

# Task 2: Adding new columns for birth year and mean left-handedness
lefthanded_data['Birth_year'] = 1986 - lefthanded_data['Age']
lefthanded_data['Mean_lh'] = lefthanded_data[['Male', 'Female']].mean(axis=1)

# Plotting mean left-handedness vs. birth year
lefthanded_data.plot(x="Birth_year", y="Mean_lh")
plt.xlabel("Birth Year")
plt.ylabel("Mean Left-handedness")
plt.show()

# Task 3: Creating a function to calculate P(LH | A)
import numpy as np

def P_lh_given_A(ages_of_death, study_year=1990):
    early_1900s_rate = lefthanded_data.iloc[-10:]['Mean_lh'].mean() / 100
    late_1900s_rate = lefthanded_data.iloc[:10]['Mean_lh'].mean() / 100
    
    P_return = np.zeros(len(ages_of_death)) 
    
    # Indexing for early 1900s
    early_mask = ages_of_death <= 1910
    P_return[early_mask] = early_1900s_rate
    
    # Indexing for late 1900s
    late_mask = ages_of_death >= 1980
    P_return[late_mask] = late_1900s_rate
    
    # Indexing for ages between 1910 and 1980
    middle_mask = np.logical_and(ages_of_death > 1910, ages_of_death < 1980)
    middle_indices = np.where(middle_mask)[0]  # Get the indices where middle_mask is True
    middle_size = len(middle_indices)  # Get the size of the middle mask
    
    
    P_return[middle_indices] = np.repeat(lefthanded_data['Mean_lh'] / 100, middle_size)
    
    return P_return



#task4
# Loading death distribution data and plotting it
data_url_2 = "https://gist.githubusercontent.com/mbonsma/2f4076aab6820ca1807f4e29f75f18ec/raw/62f3ec07514c7e31f5979beeca86f19991540796/cdc_vs00199_table310.tsv"
death_distribution_data = pd.read_csv(data_url_2, sep='\t', skiprows=[1])
death_distribution_data.dropna(subset=["Both Sexes"], inplace=True)

death_distribution_data.plot(x="Age", y="Both Sexes")
plt.xlabel("Age")
plt.ylabel("Number of People who Died")
plt.show()

#task5
# Creating a function to calculate the overall probability of left-handedness
def P_lh(death_distribution_data, study_year=1990):
    p_list = death_distribution_data["Both Sexes"] * P_lh_given_A(death_distribution_data["Age"].values, study_year)
    p = p_list.sum()
    return p / death_distribution_data["Both Sexes"].sum()

# Task 6: Creating a function to calculate P(A | LH)
def P_A_given_lh(ages_of_death, death_distribution_data, study_year=1990):
    P_A = death_distribution_data["Both Sexes"] / death_distribution_data["Both Sexes"].sum()
    P_left = P_lh(death_distribution_data, study_year)
    P_lh_A = P_lh_given_A(ages_of_death, study_year)
    
   
    P_lh_A_resized = np.append(P_lh_A, np.zeros(len(P_A) - len(P_lh_A)))  # Resize P_lh_A to match the shape of P_A
    
    return P_lh_A_resized * P_A / P_left


# Task 7: Creating a function to calculate P(A | RH)
def P_A_given_rh(ages_of_death, death_distribution_data, study_year=1990):
    P_A = death_distribution_data["Both Sexes"] / death_distribution_data["Both Sexes"].sum()
    P_right = 1 - P_lh(death_distribution_data, study_year)
    P_rh_A = 1 - P_lh_given_A(ages_of_death, study_year)
    
    
    P_rh_A_resized = np.append(P_rh_A, np.zeros(len(P_A) - len(P_rh_A)))  # Resize P_rh_A to match the shape of P_A
    
    return P_rh_A_resized * P_A / P_right


# Task 8: Plotting the probability of being a certain age at death given that you're left- or right-handed
import numpy as np

# Defining the range of ages
ages = np.arange(6, 120)  # From 6 to 119 years old, with a step size of 1 year (adjusted to match the length of left_handed_probability)

left_handed_probability = P_A_given_lh(ages, death_distribution_data)
right_handed_probability = P_A_given_rh(ages, death_distribution_data)

# Trim the probabilities to match the length of ages
left_handed_probability = left_handed_probability[:len(ages)]
right_handed_probability = right_handed_probability[:len(ages)]

plt.plot(ages, left_handed_probability, label="Left-handed")
plt.plot(ages, right_handed_probability, label="Right-handed")
plt.xlabel("Age at Death")
plt.ylabel("Probability")
plt.legend()
plt.show()




#task 9
# Calculating the mean age at death for left-handers and right-handers
average_lh_age = np.nansum(ages * left_handed_probability)
average_rh_age = np.nansum(ages * right_handed_probability)
age_difference = average_rh_age - average_lh_age

print("Average age of left-handers at death:", round(average_lh_age, 2))
print("Average age of right-handers at death:", round(average_rh_age, 2))
print("The difference in average ages is:", round(age_difference, 2), "years.")

# Task 10: Redoing the calculation for 2018
ages_2018 = np.arange(6, 120)  # Adjusted range of ages for 2018

left_handed_probability_2018 = P_A_given_lh(ages_2018, death_distribution_data, study_year=2018)
right_handed_probability_2018 = P_A_given_rh(ages_2018, death_distribution_data, study_year=2018)

# Trim the probabilities to match the length of ages_2018
left_handed_probability_2018 = left_handed_probability_2018[:len(ages_2018)]
right_handed_probability_2018 = right_handed_probability_2018[:len(ages_2018)]

# Calculate the average ages
average_lh_age_2018 = np.nansum(ages_2018 * left_handed_probability_2018)
average_rh_age_2018 = np.nansum(ages_2018 * right_handed_probability_2018)


print("The difference in average ages in 2018 is:", round(average_rh_age_2018 - average_lh_age_2018, 2), "years.")
