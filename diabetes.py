import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"



names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df = pd.read_csv(url, names = names)

array = df.values
X = array[:, 0:8]
Y = array[:,8]

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_size, random_state = seed)

model = LogisticRegression(max_iter = 500)
model.fit(X_train, Y_train)

filename = 'finalised_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)



html_temp = """<div><center><p style="font-size:14px; color:grey">&copy;2020 Anmol</p></center></div>"""

# st.markdown("""<h3>Charts (Visual)</h3>""", unsafe_allow_html= True)
# rdio2 = st.radio('',["Show", "Dont Show"], key="2")


st.markdown('''<h1><center>Diabetic ? </center></h1>''', unsafe_allow_html= True)

# im = """<div><center><img src = "https://www.kentcht.nhs.uk/wp-content/uploads/2019/08/Cardiac-Diabetes.jpg"</center></div>"""

st.write("---")
# https://cdn.vertex42.com/ExcelTemplates/Images/blood-sugar-chart-for-diabetes.png
st.markdown("""<h4 style="font-family: Arial, Helvetica, sans-serif;"><center>Use that > arrow on top left corner for all options</center></h4>""", unsafe_allow_html= True)
st.write("---")

st.markdown('''<center><h2>Select</h2>  </center>''', unsafe_allow_html=True)
choice = st.selectbox(" ", ["-----Choose-----","Facts about Diabetes", "Check it !"])
# st.markdown("""<center><h3>Stats</h3></center>""", unsafe_allow_html= True)
# rdio1 = st.radio('',["Show", "Dont Show"],key="1")
st.write("---")

if choice == "-----Choose-----":
    # main header image 
    im = """<div><center><img src = "https://www.bioworld.com/ext/resources/Stock-images/Therapeutic-topics/Diabetes/diabetes-management.png?1593552366", width = 80%></center></div>"""
    st.markdown(im, unsafe_allow_html = True)

    #First image
    im2 = """<div><center><img src = "https://www.endocrineweb.com/sites/default/files/wysiwyg_imageupload/44069/2020/09/09/Type%202%20Diabetes%20Fast%20Facts%20Graphic.png", width = 90%></center></div>"""
    #Second Image
    im3 = """<div><center><img src = "https://apollosugar.com/wp-content/uploads/2016/07/facts.jpg", width = 90%></center></div>"""
    # im2 = """"<div><center><img src = "https://cdn.vertex42.com/ExcelTemplates/Images/blood-sugar-chart-for-diabetes.png", width = 90%></center></div>"""
    st.markdown(im3, unsafe_allow_html = True)
    st.write("---")
    st.markdown(im2, unsafe_allow_html = True)
    st.write("---")

# def plots(): 
#     if rdio2 == "Show":
#         # df2 = df[:,0:8]
#         df2 = np.random.randint(100, size=50)
#         plot1 = st.selectbox("Plot between No of diabetic and non-diabetic counts",["Plot","Hide"])
#         if plot1 == "Plot":
#             plt.plot(df2)
#             plt.xlabel("Outcome")
#             plt.ylabel("Count")
#             plt.title("Number of non-diabetic(0) and diabetic(1) people in Dataset")
#             plt.show()
#         plot2 = st.selectbox("Plot between Age and Outcome(0 or 1)", ["Plot", "Hide"])
#         if plot2 == "Plot":
#             # df3 = df[:,7:8]
#             plt.hist(df2)
#             plt.xlabel("Outcome")
#             plt.ylabel("Age")
#             plt.title("Plot between average age of non-diabetic and diabetic people in Dataset")
#             plt.show()


# print(result)
# cal()
#you can also input using user

# dataset cols:
# Pregnencies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
# def cal():

elif choice == 'Facts about Diabetes':
    st.markdown('''<center><p style="font-size:19px;">10 facts on diabetes</p></center>''', unsafe_allow_html=True)
    f = ["About 422 million people worldwide have diabetes",
    "Diabetes is 1 of the leading causes of death in the world",
    "There are 2 major forms of diabetes",
    "A third type of diabetes is gestational diabetes",
    "Type 2 diabetes is much more common than type 1 diabetes",
    "People with diabetes can live long and healthy life when their diabetes is detected and well-managed",
    "Early diagnosis and intervension is the starting point for living well with diabetes",
    "The majority of diabetes deaths occur in low and middle income countries",
    "Diabetes is an important cause of blindness, amputation and kidney failure",
    "Type 2 diabetes can be prevented"]

    # Facts about diabetes

    for i in f:
        st.info(i)
        # st.write('---')
    st.markdown('''<center><p><a href = "https://www.who.int/features/factfiles/diabetes/en/", target="_blank">Source</a><p><center>''',unsafe_allow_html=True)
    st.write('---')

elif choice == 'Check it !':
    st.subheader("No of pregnancies")
    preg = st.number_input("", min_value = 0, max_value = 20)
    st.subheader("Plasma Glucose Level")
    glu = st.number_input("", min_value = 50, max_value = 500)
    st.subheader("Diastolic Blood Pressure (mm Hg)")
    bp = st.number_input("", min_value = 60, max_value = 300)
    st.subheader("Skin Thickness (mm)")
    skin = st.number_input("", min_value = 10, max_value = 60)
    st.subheader("Insulin (U/ml)")
    Insulin = st.number_input("", min_value = 40, max_value = 500)
    st.markdown("""<h4>Calculate your BMI</h4>""", unsafe_allow_html= True)
    c1, c2 = st.beta_columns(2)
    with c1:
        wt = st.number_input("Weight", min_value=30, max_value=150)
    with c2:
        ht = st.number_input("Height (cm)", min_value=145, max_value=240)
    bmi = wt / ((ht/100)**2)
    # BMI = st.number_input("BMI", min_value = 10, max_value = 30)
    #bmi can be calculated as weight / height in m2
    st.subheader(bmi)
    st.write("")
    st.subheader("Diabetes Pedigree Function")
    dpg = st.number_input("", min_value = 0.01, max_value = 2.0)
    st.subheader("Age")
    age = st.number_input("", min_value = 16, max_value = 120)

    ts = []
    ts.append(preg)
    ts.append(glu)
    ts.append(bp)
    ts.append(skin)
    ts.append(Insulin)
    ts.append(bmi)
    ts.append(dpg)
    ts.append(age)
    res = loaded_model.predict([ts])
    # print(res)
    rs = st.button("Result")
    if rs == 1 and res == 1:
        st.header("Chances of being Diabetic")
    elif rs == 1 and res == 0:
        st.header("Don't worry, NO signs of being Diabetic")
    else:
        st.header("")
    st.write('---')

# ts = [2, 120, 100, 28, 95, 20, 0.573, 49]

st.markdown(html_temp, unsafe_allow_html = True)

