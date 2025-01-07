The Behavioral Risk Factor Surveillance System (BRFSS) is a health telephone survey conducted by the Center for Disease Control and Prevention (CDC) every year. This data set comes from the information output during a survey conducted in 2015. The target variable is 0  indiacating Healthy and 1 indicating Diabetic. The data set specifically has 253,680 surveys and 21 attributes.

Attributes:-

(1) HighBP
0 = No high BP
1 = High BP

(2) HighChol
0 = No high cholesterol
1 = high cholesterol

(3) CholCheck
0 = No cholesterol check in 5 years
1 = cholesterol check done in the past 5 years

(4) Body mass index (BMI)

(5) Smoker
0 = Smoked less than a 100 cigarettes
1 = Smocked more than a 100 cigarettes


(6) Stroke
0 = Never had a stroke
1 = Has had a stroke

(7) Heart disease/attack
0 = Never had a myocardial infarction (or) coronary heart disease
1 = Has had the disease or attack.

(8) Physical activity in the past 30 days (not including job)
0 = No physical activity
1 = Yes, some physical activity.

(9) Fruits
0 = doesn’t consume 1 or more fruits per day.
1 = consumes 1 or more fruits per day.

(10) Veggies
0 = doesn’t consume vegetables 1 or more times a day.
1 = consumes vegetables 1 or more times a day.

(11) Hvy Alcohol Consump
0 = adult men NOT having more than 14 drinks per week and women NOT having more than 7
drinks per week.
1 = men having more than 14 drinks per week, women more than 7 drinks per week.

(12) AnyHealthcare
0 = Do not have any healthcare coverage, insurance or prepaid plans.
1 = Have healthcare coverage or insurance.

(13) NoDocbcCost
0 = Never had to miss seeing a doctor because of cost in the past 12 months.
1 = Has faced an instance where they couldn't see a doctor when they needed to because of cost,
in the past 12 months.

(14) GenHlth (scale of 1-5 to check how good a person’s general health is)
1 = excellent
2 = very good
3 = good
26
4 = fair
5 = poor

(15) MentHlth (including stress, depression, emotional problems, on a scale of 1-30 days, how
many days was the person’s mental health NOT good?)

(16) PhysHlth (including physical injuries and illness, on a scale of 1-30 days, for how many
days was the person’s physical health NOT good?)

(17) DiffWalk
0 = Doesn’t have a serious difficulty walking or climbing stairs.
1 = faces serious difficulty in walking/ climbing stairs.

(18) Sex
0 = female
1 = male

(19) Age (falls under the 13- level categories)
1 = 18-24 years old.
9 = 60 - 64 years old.
13 = 80 years or older.

(20) Education (on a scale of 1-6)
1 = never attended school, or attended only kindergarten .
2 = Elementary education (Grades 1 to 8)
3 = Some high school (Grades 9 to 11)
4 = High school graduate ( Grade 12 or GED)
5 = College year 1 - 3.
6 = College graduate (Year 4 or more)

(21) Income (on a scale of 1-8)
1= Less than $10,000
5 = Less than $35,000
8 = $75,000 or higher.

Out of the 253,680 data points, only 35,346 were diabetic. This gives us a ratio of 1:7 diabetic to healthy samples. If we train our data to a random sample of 50,000 data points, it will overfit for healthy, hence we must use undersampling ie. take 25,000 diabetic and 25,000 healthy samples to train our model.

We trainied 4 models, Logistic Regression, Decisoin Tree, Random Forest and SVM with 10 fold cross validation.
