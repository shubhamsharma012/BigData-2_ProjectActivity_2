import inline as inline
import matplotlib
import matplotlib_inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.simplefilter("ignore")

Test = pd.read_csv('F:\Test.csv')
Test.head()
Test.columns
Test.dtypes

print("Test dataset shape", Test.shape)

Test["Gender"].count()
Test["Gender"].value_counts()
Test["Gender"].value_counts(normalize=True)*100

Test['Gender'].value_counts(normalize=True).plot.bar(title ='Gender of loan applicant data')
print("Answer 1(a): Find out the number of male and female in loan applicants data.")
plt.xlabel('Gender')
plt.ylabel('Number of loan applicant')
plt.show()

print("Conclusion of Answer 1(a): ")
print( "Gender variable contain Male : 81% Female: 19%")

Test["Married"].count()
Test["Married"].value_counts()
Test["Married"].value_counts(normalize=True)*100

print("Total Number of people: 978")
print("Married:631")
print("Unmarried:347")

print("Answer 1(b) Find out the number of married and unmarried loan applicants.")
Test['Married'].value_counts(normalize=True).plot.bar(title='Married Status of an applicant')
plt.xlabel('Married Status')
plt.ylabel('Number of loan applicant')
plt.show()

print("Conclusion  of Answer 1(b)")
print("Number of married people : 64.519427%")
print("Number of unmarried people : 35.480573%")

Test["Self_Employed"].count()
Test["Self_Employed"].value_counts()
Test["Self_Employed"].value_counts(normalize=True)*100

print("Answer 1 (c) Find out the overall dependent status in the dataset.")
Test['Self_Employed'].value_counts(normalize=True).plot.bar(title='Dependent Status')
plt.xlabel('Dependent Status')
plt.ylabel('Number of loan applicant')
plt.show()

print("Answer 1(c) conclusion: ")
print("Among 926 people only 12.850972% are Self_Employed and rest of the 87.149028% are Not_Self_Employed")

Test["Education"].count()
Test["Education"].value_counts()
Test["Education"].value_counts(normalize=True)*100

print(" Answer 1(d) Find the count how many loan applicants are graduate and non graduate.")
Test["Education"].value_counts(normalize=True).plot.bar(title = "Education")
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

print("Answer 1(d) conclusion")
print("Total number of People : 981 ")
print("77.77778% are Graduated and 22.222222% are not Graduated")

Test["Property_Area"].count()
Test["Property_Area"].value_counts()
Test["Property_Area"].value_counts(normalize=True)*100

print("Answer 1(e) Find out the count how many loan applicants property lies in urban, rural and semi-urban areas.")
Test["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

print("Answer 1(E) conclusion")
print("35.575943% people from Semiurban area")
print("34.862385% people from Urban area")
print("29.561672% people from Rural area")

print("Answer 3")
print("To visualize and plot the distribution plot of all numerical attributes of the given train dataset i.e. ApplicantIncome,  CoApplicantIncome and LoanAmount.     ")

print("ApplicantIncome distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(Test["ApplicantIncome"]);

plt.subplot(122)
Test["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()
#
Test.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()
#
print("Coapplicant Income distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(Test["CoapplicantIncome"]);
#
plt.subplot(122)
Test["CoapplicantIncome"].plot.box(figsize=(16,5))
plt.show()
#
print("Loan Amount Variable")
plt.figure(1)
plt.subplot(121)
df=Test.dropna()
sns.distplot(df['LoanAmount']);
#
plt.subplot(122)
Test['LoanAmount'].plot.box(figsize=(16,5))
#
plt.show()
#
print("Loan Amount Term Distribution")
plt.figure(1)
plt.subplot(121)
df = Test.dropna()
sns.distplot(df["Loan_Amount_Term"]);
#
plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16,5))
plt.show()
